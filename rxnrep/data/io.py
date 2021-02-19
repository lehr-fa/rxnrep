import itertools
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from monty.serialization import loadfn
from mrnet.core.mol_entry import MoleculeEntry
from mrnet.core.reactions import Reaction as MrnetReaction

from rxnrep.core.molecule import Molecule, MoleculeError
from rxnrep.core.rdmol import create_rdkit_mol_from_mol_graph
from rxnrep.core.reaction import Reaction, ReactionError, smiles_to_reaction
from rxnrep.utils import to_path


def read_smiles_tsv_dataset(
    filename: Path, remove_H: bool, nprocs: int = 1
) -> Tuple[List[Reaction], List[bool]]:
    """
    Read reactions from dataset file.

    Args:
        filename: name of the dataset
        remove_H: whether to remove H from smiles
        nprocs:

    Returns:
        reactions: a list of rxnrep Reaction succeed in converting to dgl graphs.
            The length of this list could be shorter than the number of entries in
            the dataset file (when some entry fails).
        failed: a list of bool indicating whether each entry in the dataset file
            fails or not. The length of the list is the same as the number of
            entries in the dataset file.
    """

    filename = to_path(filename)
    df = pd.read_csv(filename, sep="\t")
    smiles_reactions = df["reaction"].tolist()

    ids = [f"{smi}_index-{i}" for i, smi in enumerate(smiles_reactions)]
    if nprocs == 1:
        reactions = [
            smiles_to_reaction_helper(smi, i, remove_H)
            for smi, i in zip(smiles_reactions, ids)
        ]
    else:
        helper = partial(smiles_to_reaction_helper, remove_H=remove_H)
        args = zip(smiles_reactions, ids)
        with multiprocessing.Pool(nprocs) as p:
            reactions = p.starmap(helper, args)

    # column names besides `reaction`
    column_names = df.columns.values.tolist()
    column_names.remove("reaction")

    succeed_reactions = []
    failed = []

    for i, rxn in enumerate(reactions):
        if rxn is None:
            failed.append(True)
        else:
            # keep other info (e.g. label) in input file as reaction property
            for name in column_names:
                rxn.set_property(name, df[name][i])

            succeed_reactions.append(rxn)
            failed.append(False)

    return succeed_reactions, failed


def read_mrnet_reaction_dataset(filename: Path, nprocs: int = 1):
    """
    Read reactions from dataset file.

    Args:
        filename: name of the dataset
        nprocs:

    Returns:
        reactions: a list of rxnrep Reaction succeed in converting to dgl graphs.
            The length of this list could be shorter than the number of entries in
            the dataset file (when some entry fails).
        failed: a list of bool indicating whether each entry in the dataset file
            fails or not. The length of the list is the same as the number of
            entries in the dataset file.
    """

    pmg_reactions = loadfn(filename)

    if nprocs == 1:
        reactions = [mrnet_rxn_helper(rxn, i) for i, rxn in enumerate(pmg_reactions)]
    else:
        ids = list(range(len(pmg_reactions)))
        args = zip(pmg_reactions, ids)
        with multiprocessing.Pool(nprocs) as p:
            reactions = p.starmap(mrnet_rxn_helper, args)

    failed = []
    succeed_reactions = []
    for rxn in reactions:
        if rxn is None:
            failed.append(True)
        else:
            failed.append(False)
            succeed_reactions.append(rxn)

    return succeed_reactions, failed


def smiles_to_reaction_helper(
    smiles_reaction: str, id: str, remove_H: bool
) -> Union[Reaction, None]:
    """
    Helper function to create reactions using multiprocessing.

    If fails, return None.
    """

    try:
        reaction = smiles_to_reaction(
            smiles_reaction,
            id=id,
            ignore_reagents=True,
            remove_H=remove_H,
            sanity_check=False,
        )
    except (MoleculeError, ReactionError):
        return None

    return reaction


def mrnet_rxn_helper(pmg_reaction: MrnetReaction, index: int) -> Union[Reaction, None]:
    """
    Helper function to process one reaction.
    """
    try:
        reaction = mrnet_reaction_to_reaction(pmg_reaction, index)
    except (MoleculeError, ReactionError):
        return None

    return reaction


def mrnet_reaction_to_reaction(pmg_reaction: MrnetReaction, index: int) -> Reaction:
    """
    Convert a pymatgen reaction to a rxnrep reaction.

    Args:
        pmg_reaction: pymatgen reaction
        index: index of the reaction in the whole dataset

    Returns:
        a rxnrep reaction
    """
    # check map numbers are the same set in all the reactants and products
    reactants_map_numbers = [mp.values() for mp in pmg_reaction.reactants_atom_mapping]
    reactants_map_numbers = set(itertools.chain.from_iterable(reactants_map_numbers))
    products_map_numbers = [mp.values() for mp in pmg_reaction.products_atom_mapping]
    products_map_numbers = set(itertools.chain.from_iterable(products_map_numbers))
    if reactants_map_numbers != products_map_numbers:
        raise ValueError(
            "Expect atom map numbers to be the same set in the reactants and products; "
            f"got {reactants_map_numbers} for the reactants and {products_map_numbers} "
            "for the products."
        )

    # convert atom map number to 1 based if there is value smaller than 1
    min_val = min(reactants_map_numbers)
    if min_val < 1:
        converter = 1 - min_val
        reactants_atom_mapping = [
            {k: v + converter for k, v in mp.items()}
            for mp in pmg_reaction.reactants_atom_mapping
        ]
        products_atom_mapping = [
            {k: v + converter for k, v in mp.items()}
            for mp in pmg_reaction.products_atom_mapping
        ]
    else:
        reactants_atom_mapping = pmg_reaction.reactants_atom_mapping
        products_atom_mapping = pmg_reaction.products_atom_mapping

    reactants = [
        mrnet_mol_entry_to_molecule(entry, mapping)
        for entry, mapping in zip(pmg_reaction.reactants, reactants_atom_mapping)
    ]

    products = [
        mrnet_mol_entry_to_molecule(entry, mapping)
        for entry, mapping in zip(pmg_reaction.products, products_atom_mapping)
    ]

    reactant_ids = "+".join([str(i) for i in pmg_reaction.reactant_ids])
    product_ids = "+".join([str(i) for i in pmg_reaction.product_ids])
    reaction_id = f"{reactant_ids}->{product_ids}_index-{index}"

    # additional property
    free_energy = sum([m.get_property("free_energy") for m in products]) - sum(
        [m.get_property("free_energy") for m in reactants]
    )

    reaction = Reaction(
        reactants,
        products,
        id=reaction_id,
        sanity_check=False,
        properties={"free_energy": free_energy},
    )

    return reaction


def mrnet_mol_entry_to_molecule(
    mol_entry: MoleculeEntry, atom_mapping: Dict[int, int]
) -> Molecule:
    """
    Convert a pymatgen molecule entry to a rxnrep molecule, and set the atom mapping.

    Args:
        mol_entry: molecule entry
        atom_mapping: {atom_index: map_number} atom map number for atoms in the molecule

    Returns:
        rxnrep molecule with atom mapping
    """

    sorted_atom_index = sorted(atom_mapping.keys())

    # check has atom map number for each atom
    if sorted_atom_index != list(range(mol_entry.num_atoms)):
        raise ValueError(
            f"Incorrect `atom_mapping`. Molecule has {mol_entry.num_atoms} atoms, "
            f"but provided mapping number are for atoms {sorted_atom_index}."
        )

    rdkit_mol, _ = create_rdkit_mol_from_mol_graph(mol_entry.mol_graph)

    # set rdkit mol atom map number
    for i in sorted_atom_index:
        map_number = atom_mapping[i]
        rdkit_mol.GetAtomWithIdx(i).SetAtomMapNum(map_number)

    # NOTE, The total charge of the created rdkit molecule may be different from the
    # charge of the pymatgen Molecule. Currently, there is not a good method to
    # handle metals. We set the charge of Molecule to what is in pymatgen Molecule
    # explicitly.
    mol = Molecule(
        rdkit_mol,
        id=mol_entry.entry_id,
        properties={"free_energy": mol_entry.get_free_energy()},
    )
    mol.charge = mol_entry.molecule.charge

    return mol
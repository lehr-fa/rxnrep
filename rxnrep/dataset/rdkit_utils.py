import multiprocessing
import numpy as np
from rdkit import Chem
import copy
from typing import List, Tuple, Dict, Union, Set


def check_molecule_atom_mapped(m: Chem.Mol) -> bool:
    """
    Check if all atoms in a molecule are mapped.

    Args:
        m: rdkit mol
    """
    for atom in m.GetAtoms():
        if not atom.HasProp("molAtomMapNumber"):
            return False

    return True


def check_reaction_atom_mapped(reaction: str) -> Tuple[bool, bool]:
    """
    Check both reactants and products of a reaction are atom mapped.

    Args:
        reaction: smiles representation of a reaction.

    Returns:
        reactants_mapped: whether reactants is mapped, `None` if molecule cannot be
            created.
        products_mapped: whether products is mapped, `None` if molecule cannot be created.
    """
    reactants, _, products = reaction.strip().split(">")
    rct = Chem.MolFromSmiles(reactants)
    prdt = Chem.MolFromSmiles(products)

    if rct is None or prdt is None:
        return None, None

    else:
        reactants_mapped = check_molecule_atom_mapped(rct)
        products_mapped = check_molecule_atom_mapped(prdt)

        return reactants_mapped, products_mapped


def check_all_reactions_atom_mapped(
    reactions: List[str], nprocs=1, print_result=False
) -> List[Tuple[bool, bool]]:
    """
    Check that reactants and products in all reactions are atom mapped.

    Args:
        reactions: list of smiles representation of a reaction
        nprocs: number of processes to use
        print_result: whether to print out results

    Returns:
        mapped: mapped[i] is a two-tuple indicating whether the reactants and products
        are mapped.
    """
    if nprocs == 1:
        mapped = [check_reaction_atom_mapped(rxn) for rxn in reactions]
    else:
        with multiprocessing.Pool(nprocs) as p:
            mapped = p.map(check_reaction_atom_mapped, reactions)

    if print_result:
        if np.all(mapped):
            print("Reactants and products in all reactions are mapped.")
        else:
            n_failed = 0
            n_rct = 0
            n_prdt = 0
            for i, mp in enumerate(mapped):

                if mp[0] is None or mp[1] is None:
                    n_failed += 1
                elif not mp[0] and not mp[1]:
                    n_rct += 1
                    n_prdt += 1
                    print(f"{i} both reactants and products are not mapped")
                elif not mp[0]:
                    n_rct += 1
                    print(f"{i} reactants are not mapped")
                elif not mp[1]:
                    n_prdt += 1
                    print(f"{i} reactants are not mapped")
            print(
                f"Total number of reactions: {len(mapped)}; reactants not mapped: "
                f"{n_rct}; products not mapped {n_prdt}; "
                f"molecules cannot be converted: {n_failed}."
            )

        print("Done!")

    return mapped


def check_bonds_mapped(m: Chem.Mol) -> Tuple[bool, bool]:
    """
    The the mappings of the bonds in a moleucle.

    Returns:
        has_bond_both_atoms_not_mapped
        has_bond_one_atom_not_mapped
    """
    has_bond_both_atoms_not_mapped = False
    has_bond_one_atom_not_mapped = False
    for bond in m.GetBonds():
        atom1_mapped = bond.GetBeginAtom().HasProp("molAtomMapNumber")
        atom2_mapped = bond.GetEndAtom().HasProp("molAtomMapNumber")
        if has_bond_both_atoms_not_mapped and has_bond_one_atom_not_mapped:
            break
        # both not mapped
        if not atom1_mapped and not atom2_mapped:
            has_bond_both_atoms_not_mapped = True
        # one mapped, the other not
        elif atom1_mapped != atom2_mapped:
            has_bond_one_atom_not_mapped = True

    return has_bond_both_atoms_not_mapped, has_bond_one_atom_not_mapped


def check_reaction_bonds_mapped(reaction: str) -> Tuple[bool, bool]:
    """
    Check the atom mapping for bonds in the reactants of a reaction.

    Args:
        reaction: smiles representation of a reaction.

    Returns:
        has_bond_both_atoms_not_mapped
        has_bond_one_atom_not_mapped
    """
    reactants, _, products = reaction.strip().split(">")
    rct = Chem.MolFromSmiles(reactants)
    prdt = Chem.MolFromSmiles(products)

    if rct is None or prdt is None:
        return None, None

    else:
        has_bonds_both_not_mapped, has_bonds_one_not_mapped = check_bonds_mapped(rct)

        return has_bonds_both_not_mapped, has_bonds_one_not_mapped


def check_all_reactions_bonds_mapped(
    reactions: List[str], nprocs=1, print_result=False
) -> List[Tuple[bool, bool]]:
    if nprocs == 1:
        mapped = [check_reaction_bonds_mapped(rxn) for rxn in reactions]
    else:
        with multiprocessing.Pool(nprocs) as p:
            mapped = p.map(check_reaction_bonds_mapped, reactions)

    if print_result:
        if np.all(mapped):
            print("All bonds are mapped.")
        else:
            n_failed = 0
            n_both = 0
            n_one = 0
            for i, mp in enumerate(mapped):

                if mp[0] is None or mp[1] is None:
                    n_failed += 1
                elif mp[0] and mp[1]:
                    n_both += 1
                    print(f"{i} has bonds both atoms not mapped")
                elif mp[0] != mp[1]:
                    n_one += 1
                    print(f"{i} has bonds one atom not mapped")
            print(
                f"Total number of reactions: {len(mapped)}; reactions having bonds "
                f"both atoms not mapped: {n_both}; having bonds one atom not mapped "
                f" {n_one}; molecules cannot be converted: {n_failed}."
            )

        print("Done!")

    return mapped


def canonicalize_smiles_reaction(
    reaction: str,
) -> Tuple[Union[str, None], Union[None, str]]:
    """
    Canonicalize a smiles reaction to make reactants and products have the same
    composition.

    This ensures the reactants and products have the same composition, achieved in the
    below steps:

    1. remove reactant molecules from reactants none of their atoms are present in
       the products
    2. adjust atom mapping between reactants and products and add atom mapping number
       for reactant atoms without a mapping number (although there is no corresponding
       atom in the products)
    3. create new products by editing the reactants: removing bonds in the reactants
       but not in the products and adding bonds not in the reactants but in the products

    Args:
        reaction: smiles representation of a reaction

    Returns:
        reaction: canonicalized smiles reaction, `None` if canonicalize failed
        error: error message, `None` if canonicalize succeed
    """

    # Step 1, adjust reagents
    try:
        rxn_smi = adjust_reagents(reaction)
    except (MoleculeCreationError, AtomMapNumberError) as e:
        return None, str(e).rstrip()

    # Step 2, adjust atom mapping
    try:
        reactants_smi, reagents_smi, products_smi = rxn_smi.strip().split(">")
        reactants = Chem.MolFromSmiles(reactants_smi)
        products = Chem.MolFromSmiles(products_smi)
        reactants, products = adjust_atom_map_number(reactants, products)
    except AtomMapNumberError as e:
        return None, str(e).rstrip()

    # Step 3, create new products
    try:
        bond_changes = get_reaction_bond_change(reactants, products)
        new_products = edit_molecule(reactants, bond_changes)
    except MoleculeCreationError as e:
        return None, str(e).rstrip()

    # write canonicalized reaction to smiles
    reactants_smi = Chem.MolToSmiles(set_all_H_to_explicit(reactants))
    products_smi = Chem.MolToSmiles(set_all_H_to_explicit(new_products))
    canoical_reaction = ">".join([reactants_smi, reagents_smi, products_smi])

    return canoical_reaction, None


def get_mol_atom_mapping(m: Chem.Mol) -> List[int]:
    """
    Get atom mapping for an rdkit molecule.

    Args:
        m: rdkit molecule

    Returns:
         atom mapping for each atom. `None` if the atom is not mapped.
    """
    mapping = []
    for atom in m.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            mapping.append(atom.GetAtomMapNum())
        else:
            mapping.append(None)
    return mapping


def adjust_reagents(reaction: str) -> str:
    """
    Move reagents in the reactants or products to the reagents collection.

    For a smiles reaction of the type `aaa>bbb>ccc`, aaa is the reactants, bbb is the
    reagents, and ccc is the products. It could happen that some molecule in aaa (ccc)
    does not have a single atom in ccc (aaa). Such molecules should actually be reagents.
    This function moves such molecules from aaa (ccc) to bbb.

    Args:
        reaction: smiles representation of an atom mapped reaction

    Returns:
         smiles reaction with the place of reagents adjusted
    """

    reactants_smi, reagents_smi, products_smi = reaction.strip().split(">")

    reactants = [Chem.MolFromSmiles(s) for s in reactants_smi.split(".")]
    products = [Chem.MolFromSmiles(s) for s in products_smi.split(".")]
    if None in reactants or None in products:
        raise MoleculeCreationError(f"Cannot create molecules from: {reaction}")

    # get atom mapping
    mapping_rcts = [set(get_mol_atom_mapping(m)) for m in reactants]
    mapping_prdts = [set(get_mol_atom_mapping(m)) for m in products]
    mapping_rcts_all = set()
    mapping_prdts_all = set()
    for x in mapping_rcts:
        mapping_rcts_all.update(x)
    for x in mapping_prdts:
        mapping_prdts_all.update(x)
    if None in mapping_prdts_all:
        raise AtomMapNumberError("Products has atom without map number.")

    new_reactants = []
    new_reagents = []
    new_products = []
    # move reactant to reagent if none of its atoms is in the product
    for i, mapping in enumerate(mapping_rcts):
        if len(mapping & mapping_prdts_all) == 0:
            new_reagents.append(reactants[i])
        else:
            new_reactants.append(reactants[i])

    # move product to reagent if none of its atoms is in the reactant
    for i, mapping in enumerate(mapping_prdts):
        if len(mapping & mapping_rcts_all) == 0:
            new_reagents.append(products[i])
        else:
            new_products.append(products[i])

    # remove atom mapping in new reagents
    for m in new_reagents:
        for a in m.GetAtoms():
            a.ClearProp("molAtomMapNumber")

    reactants_smi = ".".join([Chem.MolToSmiles(m) for m in new_reactants])
    products_smi = ".".join([Chem.MolToSmiles(m) for m in new_products])
    reagents_smi = ".".join([reagents_smi] + [Chem.MolToSmiles(m) for m in new_reagents])

    reaction = ">".join([reactants_smi, reagents_smi, products_smi])

    return reaction


def adjust_atom_map_number(
    reactant: Chem.Mol, product: Chem.Mol
) -> Tuple[Chem.Mol, Chem.Mol]:
    """
    Adjust atom map number between the reactant and product.

    The below steps are performed:

    1. Check the map numbers are unique for the reactants (products), i.e. each map
       number only occur once such that no map number is associated with two atoms.
    2. Check whether all product atoms are mapped to reactant atoms. If not, throw an
       error. It is not required that all reactant atoms should be mapped.
    3. Renumber the existing atom map numbers to let it start from 1 and be consecutive.
    4. Add atom map numbers to atoms in the reactant if they do not have one.

    The input reactant and product are not be modified.

    Args:
        reactant: rdkit molecule
        product: rdkit molecule

    Returns:
        reactant and product with atom map number adjusted
    """

    reactant = copy.deepcopy(reactant)
    product = copy.deepcopy(product)

    rct_mapping = get_mol_atom_mapping(reactant)
    prdt_mapping = get_mol_atom_mapping(product)

    # Step 1, check map number uniqueness
    if None in prdt_mapping:
        raise AtomMapNumberError("Products has atom without map number.")
    if len(prdt_mapping) != len(set(prdt_mapping)):
        raise AtomMapNumberError("Products has atoms with the same map number.")
    rct_mapping_no_None = [mp for mp in rct_mapping if mp is not None]
    if len(rct_mapping_no_None) != len(set(rct_mapping_no_None)):
        raise AtomMapNumberError("Reactants has atoms with the same map number.")

    # Step 2, check all product atoms are mapped
    if not set(prdt_mapping).issubset(set(rct_mapping)):
        raise AtomMapNumberError("Products has atom not mapped to product.")

    # Step 3, Renumber existing atom map

    # clear reactant atom map number
    for a in reactant.GetAtoms():
        a.ClearProp("molAtomMapNumber")

    # set atom map number
    for i, mp in enumerate(prdt_mapping):
        rct_atom_idx = rct_mapping.index(mp)
        prdt_atom_idx = i
        reactant.GetAtomWithIdx(rct_atom_idx).SetAtomMapNum(i + 1)
        product.GetAtomWithIdx(prdt_atom_idx).SetAtomMapNum(i + 1)

    # Step 4, add atom map number for reactant atoms does not have one
    i = len(prdt_mapping)
    for a in reactant.GetAtoms():
        if not a.HasProp("molAtomMapNumber"):
            a.SetAtomMapNum(i + 1)
            i += 1

    return reactant, product


def get_reaction_bond_change(
    reactant: Chem.Mol, product: Chem.Mol, use_mapped_atom_index=False
) -> Set[Tuple[int, int, float]]:
    """
    Get the changes of the bonds to make the products from the reactants.

    Args:
        reactant: rdkit molecule
        product: rdkit molecule
        use_mapped_atom_index: this determines what to use for the atom index in the
            returned bond changes. If `False`, using the atom index in the underlying
            rdkit molecule; if `True`, using the mapped atom index.

    Returns:
        bond_change: each element is a three-tuple (atom_1, atom_2, change_type) denoting
            the change of a bond. `atom_1` and `atom_2` are indices of the two atoms
            forming the bond. The atom indices could either be the non-mapped or the
            mapped indices, depending on `use_mapped_atom_index`.
            `change_type` can take 0, 1, 2, 3, and 1.5, meaning losing a bond, forming
            a single, double, triple, and aromatic bond, respectively.
    """

    # bonds in reactant (only consider bonds whose atoms are mapped)
    bonds_rct = {}
    for bond in reactant.GetBonds():
        bond_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
        if bond_atoms[0].HasProp("molAtomMapNumber") and bond_atoms[1].HasProp(
            "molAtomMapNumber"
        ):
            num_pair = tuple(sorted([a.GetAtomMapNum() for a in bond_atoms]))
            bonds_rct[num_pair] = bond.GetBondTypeAsDouble()

    # bonds in product (only consider bonds whose atoms are mapped)
    bonds_prdt = {}
    for bond in product.GetBonds():
        bond_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
        if bond_atoms[0].HasProp("molAtomMapNumber") and bond_atoms[1].HasProp(
            "molAtomMapNumber"
        ):
            num_pair = tuple(sorted([a.GetAtomMapNum() for a in bond_atoms]))
            bonds_prdt[num_pair] = bond.GetBondTypeAsDouble()

    bond_changes = set()

    for bond in bonds_rct:
        if bond not in bonds_prdt:
            # lost bond
            bond_changes.add((bond[0], bond[1], 0.0))
        else:
            if bonds_rct[bond] != bonds_prdt[bond]:
                # changed bond
                bond_changes.add((bond[0], bond[1], bonds_prdt[bond]))

    for bond in bonds_prdt:
        if bond not in bonds_rct:
            # new bond
            bond_changes.add((bond[0], bond[1], bonds_prdt[bond]))

    # convert mapped atom index to the underlying rdkit atom index (non-mapped)
    # of the reactant
    if not use_mapped_atom_index:
        atom_mp = get_mol_atom_mapping(reactant)
        converter = {v: i for i, v in enumerate(atom_mp) if v is not None}
        bond_changes_new_atom_index = []
        for atom1, atom2, change in bond_changes:
            idx1, idx2 = sorted([converter[atom1], converter[atom2]])
            bond_changes_new_atom_index.append((idx1, idx2, change))

        bond_changes = set(bond_changes_new_atom_index)

    return bond_changes


def edit_molecule(mol: Chem.Mol, edits: Set[Tuple[int, int, float]]) -> Chem.Mol:
    """
    Edit a molecule to generate a new one by applying the bond changes.

    Args:
        mol: rdkit molecule to edit
        edits: each element is a three-tuple (atom_1, atom_2, change_type) denoting
            the change of a bond. `atom_1` and `atom_2` are indices of the two atoms
            forming the bond, and `change_type` can take 0, 1, 2, 3, and 1.5, meaning
            losing a bond, forming a single, double, triple, and aromatic bond,
            respectively.

    Returns:
        new_mol: a new molecule after applying the bond edits to the input molecule
    """

    bond_change_to_type = {
        1.0: Chem.rdchem.BondType.SINGLE,
        2.0: Chem.rdchem.BondType.DOUBLE,
        3.0: Chem.rdchem.BondType.TRIPLE,
        1.5: Chem.rdchem.BondType.AROMATIC,
    }

    # Let all explicit H be implicit. This increases the number of implicit H and
    # allows the adjustment of the number of implicit H to satisfy valence rule
    mol = set_all_H_to_implicit(mol)

    rw_mol = Chem.RWMol(mol)

    for atom1, atom2, change_type in edits:
        bond = rw_mol.GetBondBetweenAtoms(atom1, atom2)
        if bond is not None:
            rw_mol.RemoveBond(atom1, atom2)
        if change_type > 0:
            rw_mol.AddBond(atom1, atom2, bond_change_to_type[change_type])

    new_mol = rw_mol.GetMol()

    # After editing, we set all hydrogen to explicit
    new_mol = set_all_H_to_explicit(new_mol)

    # check validity of created new mol by writing to smile and read back (sanity check
    # is performed when reading it back)
    new_mol_smi = Chem.MolToSmiles(new_mol)
    new_mol = Chem.MolFromSmiles(new_mol_smi)
    if new_mol is None:
        raise MoleculeCreationError("Cannot get correct mol after editing")

    return new_mol


def set_all_H_to_implicit(m: Chem.Mol) -> Chem.Mol:
    """
    Set all the hydrogens on atoms to implicit.


    Args:
        m: rdkit molecule

    Returns rdkit molecule with all hydrogens implicit
    """
    m2 = Chem.RemoveHs(m, implicitOnly=False)
    for atom in m2.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    Chem.SanitizeMol(m2)

    return m2


def set_all_H_to_explicit(m: Chem.Mol) -> Chem.Mol:
    """
    Set all the hydrogens on atoms to explicit.


    Args:
        m: rdkit molecule

    Returns rdkit molecule with all hydrogens explicit
    """

    # method 1
    # m2 = Chem.RemoveHs(m, implicitOnly=False)
    # for atom in m2.GetAtoms():
    #     num_H = atom.GetTotalNumHs()
    #     atom.SetNoImplicit(True)
    #     atom.SetNumExplicitHs(num_H)

    # method 2
    m2 = Chem.AddHs(m, explicitOnly=False)
    for atom in m2.GetAtoms():
        atom.SetNoImplicit(True)
    m2 = Chem.RemoveHs(m2, implicitOnly=False)
    Chem.SanitizeMol(m2)

    return m2


def get_reaction_atom_mapping(
    reactants: List[Chem.Mol], products: List[Chem.Mol]
) -> List[Dict[int, Tuple[int, int]]]:
    """
    Create atom mapping between reactants and products.

    Each dictionary is the mapping for a reactant, in the format:
     {atom_index: {product_index, product_atom_index}}.

    If a mapping cannot be found for an atom in a reactant molecule, it is set to (
    None, None).
    """
    reactants_mp = [get_mol_atom_mapping(m) for m in reactants]
    products_mp = [get_mol_atom_mapping(m) for m in products]

    mappings = []
    for rct_mp in reactants_mp:
        molecule = {}
        for atom_idx, atom_mp in enumerate(rct_mp):
            for prdt_idx, prdt_mp in enumerate(products_mp):
                if atom_mp in prdt_mp:
                    idx = (prdt_idx, prdt_mp.index(atom_mp))
                    break
            else:
                idx = (None, None)
            molecule[atom_idx] = idx

        mappings.append(molecule)

    return mappings


class MoleculeCreationError(Exception):
    def __init__(self, msg):
        super(MoleculeCreationError, self).__init__(msg)
        self.msg = msg


class AtomMapNumberError(Exception):
    def __init__(self, msg):
        super(AtomMapNumberError, self).__init__(msg)
        self.msg = msg

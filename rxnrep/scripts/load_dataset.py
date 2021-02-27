import warnings
from pathlib import Path

from torch.utils.data import DataLoader

from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.data.green import GreenDataset


def load_Green_dataset(args):

    state_dict_filename = get_state_dict_filename(args)

    # adjust args controlling labels
    max_hop_distance = args.max_hop_distance if "max_hop_distance" in args else None
    atom_type_masker_ratio = (
        args.atom_type_masker_ratio if "atom_type_masker_ratio" in args else None
    )
    atom_type_masker_use_masker_value = (
        args.atom_type_masker_use_masker_value
        if "atom_type_masker_use_masker_value" in args
        else None
    )
    if "have_activation_energy_ratio" in args:
        have_activation_energy_ratio_trainset = args.have_activation_energy_ratio
        have_activation_energy_ratio_val_test_set = 1.0
    else:
        have_activation_energy_ratio_trainset = None
        have_activation_energy_ratio_val_test_set = None

    atom_featurizer_kwargs = {
        "atom_total_degree_one_hot": {"allowable_set": list(range(5))},
        "atom_total_valence_one_hot": {"allowable_set": list(range(5))},
        "atom_num_radical_electrons_one_hot": {"allowable_set": list(range(3))},
    }

    trainset = GreenDataset(
        filename=args.trainset_filename,
        atom_featurizer=AtomFeaturizer(featurizer_kwargs=atom_featurizer_kwargs),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=state_dict_filename,
        num_processes=args.nprocs,
        # label args
        max_hop_distance=max_hop_distance,
        atom_type_masker_ratio=atom_type_masker_ratio,
        atom_type_masker_use_masker_value=atom_type_masker_use_masker_value,
        have_activation_energy_ratio=have_activation_energy_ratio_trainset,
    )

    state_dict = trainset.state_dict()

    valset = GreenDataset(
        filename=args.valset_filename,
        atom_featurizer=AtomFeaturizer(featurizer_kwargs=atom_featurizer_kwargs),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
        # label args
        max_hop_distance=max_hop_distance,
        atom_type_masker_ratio=atom_type_masker_ratio,
        atom_type_masker_use_masker_value=atom_type_masker_use_masker_value,
        have_activation_energy_ratio=have_activation_energy_ratio_val_test_set,
    )

    testset = GreenDataset(
        filename=args.testset_filename,
        atom_featurizer=AtomFeaturizer(featurizer_kwargs=atom_featurizer_kwargs),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
        # label args
        max_hop_distance=max_hop_distance,
        atom_type_masker_ratio=atom_type_masker_ratio,
        atom_type_masker_use_masker_value=atom_type_masker_use_masker_value,
        have_activation_energy_ratio=have_activation_energy_ratio_val_test_set,
    )

    # save dataset state dict for retraining or prediction
    trainset.save_state_dict_file(args.dataset_state_dict_filename)
    print(
        "Trainset size: {}, valset size: {}: testset size: {}.".format(
            len(trainset), len(valset), len(testset)
        )
    )

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=trainset.collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valset.collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=testset.collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # Add dataset state dict to args to log it
    args.dataset_state_dict = state_dict

    # Add info that will be used in the model to args for easy access
    args.feature_size = trainset.feature_size
    args.label_mean = trainset.label_mean
    args.label_std = trainset.label_std

    if max_hop_distance is not None:
        class_weight = trainset.get_class_weight()
        args.atom_hop_dist_class_weight = class_weight["atom_hop_dist"]
        args.bond_hop_dist_class_weight = class_weight["bond_hop_dist"]
        args.atom_hop_dist_num_classes = len(args.atom_hop_dist_class_weight)
        args.bond_hop_dist_num_classes = len(args.bond_hop_dist_class_weight)
    if atom_type_masker_ratio is not None:
        args.masked_atom_type_num_classes = len(trainset.get_species())

    return train_loader, val_loader, test_loader


def get_state_dict_filename(args):
    """
    Check dataset state dict if in restore mode
    """

    if args.restore:
        if args.dataset_state_dict_filename is None:
            warnings.warn(
                "Restore with `args.dataset_state_dict_filename` set to None."
            )
            state_dict_filename = None
        elif not Path(args.dataset_state_dict_filename).exists():
            warnings.warn(
                f"args.dataset_state_dict_filename: `{args.dataset_state_dict_filename} "
                "not found; set to `None`."
            )
            state_dict_filename = None
        else:
            state_dict_filename = args.dataset_state_dict_filename
    else:
        state_dict_filename = None

    return state_dict_filename
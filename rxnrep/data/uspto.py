import logging
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import torch
from sklearn.utils import class_weight

from rxnrep.data.datamodule import (
    BaseClassificationDataModule,
    BaseContrastiveDataModule,
    BaseMorganDataModule,
)
from rxnrep.data.dataset import (
    BaseContrastiveDataset,
    BaseLabelledDataset,
    ClassicalFeatureDataset,
)
from rxnrep.data.featurizer import (
    AtomFeaturizer,
    BondFeaturizer,
    GlobalFeaturizer,
    MorganFeaturizer,
)
from rxnrep.data.inout import read_smiles_tsv_dataset

logger = logging.getLogger(__name__)


def read_uspto_file(filename: Path, nprocs):
    logger.info("Start reading dataset ...")

    succeed_reactions, failed = read_smiles_tsv_dataset(
        filename, remove_H=True, nprocs=nprocs
    )

    counter = Counter(failed)
    logger.info(
        f"Finish reading dataset. Number succeed {counter[False]}, "
        f"number failed {counter[True]}."
    )

    return succeed_reactions, failed


class USPTODataset(BaseLabelledDataset):
    """
    USPTO dataset.

    Args:
        has_class_label: whether the dataset provides class label. The Schneider dataset
            has.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        atom_featurizer: Callable,
        bond_featurizer: Callable,
        global_featurizer: Callable,
        *,
        build_reaction_graph=True,
        init_state_dict: Optional[Union[Dict, Path]] = None,
        transform_features: bool = True,
        return_index: bool = True,
        num_processes: int = 1,
        #
        # args to control labels
        #
        has_class_label: bool = False,
    ):
        self.has_class_label = has_class_label

        super().__init__(
            filename,
            atom_featurizer,
            bond_featurizer,
            global_featurizer,
            build_reaction_graph=build_reaction_graph,
            init_state_dict=init_state_dict,
            transform_features=transform_features,
            return_index=return_index,
            num_processes=num_processes,
        )

    def read_file(self, filename: Path):
        return read_uspto_file(filename, self.nprocs)

    def generate_labels(self):
        """
        Create `reaction_type` label.
        """
        if self.has_class_label:
            for i, rxn in enumerate(self.reactions):
                rxn_class = rxn.get_property("reaction_type")
                self.labels[i]["reaction_type"] = torch.as_tensor(
                    [int(rxn_class)], dtype=torch.int64
                )

    def get_class_weight(
        self, num_reaction_classes: int = None, class_weight_as_1: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Create class weight to be used in cross entropy losses.

        Args:
            num_reaction_classes: number of reaction classes in the dataset. The class
            labels should be 0, 1, 2, ... num_reaction_classes-1.
            class_weight_as_1: If `True`, the weight for all classes is set to 1.0;
                otherwise, it is inversely proportional to the number of data points in
                the dataset
        """

        if self.has_class_label:
            if class_weight_as_1:
                w = torch.ones(num_reaction_classes)
            else:
                rxn_classes = [rxn.get_property("label") for rxn in self.reactions]

                # class weight for reaction classes
                w = class_weight.compute_class_weight(
                    "balanced",
                    classes=list(range(num_reaction_classes)),
                    y=rxn_classes,
                )
                w = torch.as_tensor(w, dtype=torch.float32)

            weight = {"reaction_type": w}
        else:
            weight = {}

        return weight


class USPTOContrastiveDataset(BaseContrastiveDataset):
    def read_file(self, filename: Path):
        return read_uspto_file(filename, self.nprocs)


class USPTOClassicalFeaturesDataset(ClassicalFeatureDataset):
    def read_file(self, filename: Path):
        return read_uspto_file(filename, self.nprocs)

    def generate_labels(self):
        """
        Labels for all reactions.

        Add `reaction_type`.
        """

        labels = []
        for i, rxn in enumerate(self.reactions):
            rxn_class = rxn.get_property("reaction_type")
            labels.append(
                {"reaction_type": torch.as_tensor(int(rxn_class), dtype=torch.int64)}
            )
        return labels

    def get_class_weight(
        self, num_reaction_classes: int = None, class_weight_as_1: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Create class weight to be used in cross entropy losses.

        Args:
            num_reaction_classes: number of reaction classes in the dataset. The class
            labels should be 0, 1, 2, ... num_reaction_classes-1.
            class_weight_as_1: If `True`, the weight for all classes is set to 1.0;
                otherwise, it is inversely proportional to the number of data points in
                the dataset
        """

        if class_weight_as_1:
            w = torch.ones(num_reaction_classes)
        else:
            rxn_classes = [rxn.get_property("label") for rxn in self.reactions]

            # class weight for reaction classes
            w = class_weight.compute_class_weight(
                "balanced",
                classes=list(range(num_reaction_classes)),
                y=rxn_classes,
            )
            w = torch.as_tensor(w, dtype=torch.float32)

        weight = {"reaction_type": w}

        return weight


class UsptoClassificationDataModule(BaseClassificationDataModule):
    """
    Uspto classification data module.
    """

    def setup(self, stage: Optional[str] = None):

        init_state_dict = self.get_init_state_dict()

        has_class_label = self.num_reaction_classes is not None

        atom_featurizer = AtomFeaturizer()
        bond_featurizer = BondFeaturizer()
        global_featurizer = GlobalFeaturizer()

        self.data_train = USPTODataset(
            filename=self.trainset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=init_state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            # label args
            has_class_label=has_class_label,
        )

        state_dict = self.data_train.state_dict()

        self.data_val = USPTODataset(
            filename=self.valset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            # label args
            has_class_label=has_class_label,
        )

        self.data_test = USPTODataset(
            filename=self.testset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            # label args
            has_class_label=has_class_label,
        )

        # save dataset state dict
        self.data_train.save_state_dict_file(self.state_dict_filename)

        logger.info(
            f"Trainset size: {len(self.data_train)}, valset size: {len(self.data_val)}: "
            f"testset size: {len(self.data_test)}."
        )


class UsptoContrastiveDataModule(BaseContrastiveDataModule):
    """
    Uspto datamodule for contrastive learning.
    """

    def setup(self, stage: Optional[str] = None):

        init_state_dict = self.get_init_state_dict()

        atom_featurizer = AtomFeaturizer()
        bond_featurizer = BondFeaturizer()
        global_featurizer = GlobalFeaturizer()

        self.data_train = USPTOContrastiveDataset(
            filename=self.trainset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=init_state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            transform1=self.transform1,
            transform2=self.transform2,
        )

        state_dict = self.data_train.state_dict()

        self.data_val = USPTOContrastiveDataset(
            filename=self.valset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            transform1=self.transform1,
            transform2=self.transform2,
        )

        self.data_test = USPTOContrastiveDataset(
            filename=self.testset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            transform1=self.transform1,
            transform2=self.transform2,
        )

        # save dataset state dict
        self.data_train.save_state_dict_file(self.state_dict_filename)

        logger.info(
            f"Trainset size: {len(self.data_train)}, valset size: {len(self.data_val)}: "
            f"testset size: {len(self.data_test)}."
        )


class UsptoMorganDataModule(BaseMorganDataModule):
    """
    Uspto datamodule using Morgan feats for classification.
    """

    def setup(self, stage: Optional[str] = None):

        featurizer = MorganFeaturizer(
            radius=self.morgan_radius,
            size=self.morgan_size,
        )

        self.data_train = USPTOClassicalFeaturesDataset(
            filename=self.trainset_filename,
            featurizer=featurizer,
            feature_type=self.feature_combine_method,
            num_processes=self.num_processes,
        )

        self.data_val = USPTOClassicalFeaturesDataset(
            filename=self.valset_filename,
            featurizer=featurizer,
            feature_type=self.feature_combine_method,
            num_processes=self.num_processes,
        )

        self.data_test = USPTOClassicalFeaturesDataset(
            filename=self.testset_filename,
            featurizer=featurizer,
            feature_type=self.feature_combine_method,
            num_processes=self.num_processes,
        )

        logger.info(
            f"Trainset size: {len(self.data_train)}, valset size: {len(self.data_val)}: "
            f"testset size: {len(self.data_test)}."
        )

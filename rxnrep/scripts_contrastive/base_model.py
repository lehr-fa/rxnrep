"""
Base model for Constrative representation learning.
"""

from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler

from rxnrep.scripts.utils import TimeMeter


class BaseLightningModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        # save params to be accessible via self.hparams
        self.save_hyperparameters(params)

        self.model = self.create_model(self.hparams)

        self.classification_tasks = {}
        self.regression_tasks = {}
        self.init_tasks()

        self.metrics = self.init_metrics()

        self.timer = TimeMeter()

    def forward(self, batch, returns: str = "reaction_feature"):
        """
        Args:
            batch:
            returns: the type of features (embeddings) to return. Optionals are
                `reaction_feature`, 'diff_feature_before_rxn_conv',
                and 'diff_feature_after_rxn_conv'.

        Returns:
            If returns = `reaction_feature`, return a 2D tensor of reaction features,
            each row for a reaction;
            If returns = `diff_feature_before_rxn_conv` or `diff_feature_after_rxn_conv`,
            return a dictionary of atom, bond, and global features.
            As the name suggests, the returned features can be `before` or `after`
            the reaction conv layers.
            If returns = `activation_energy` (`reaction_energy`), return the activation
            (reaction) energy predicted by the decoder.
        """
        indices, mol_graphs, rxn_graphs, labels, metadata = batch

        # lightning cannot move dgl graphs to gpu, so do it manually
        mol_graphs = mol_graphs.to(self.device)
        rxn_graphs = rxn_graphs.to(self.device)

        nodes = ["atom", "global"]
        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}
        feats["bond"] = mol_graphs.edges["bond"].data.pop("feat")

        if returns == "reaction_feature":
            _, reaction_feats = self.model(mol_graphs, rxn_graphs, feats, metadata)
            return reaction_feats

        elif returns == "diff_feature_after_rxn_conv":
            diff_feats, _ = self.model(mol_graphs, rxn_graphs, feats, metadata)
            return diff_feats

        elif returns == "diff_feature_before_rxn_conv":
            diff_feats = self.model.get_diff_feats(
                mol_graphs, rxn_graphs, feats, metadata
            )
            return diff_feats
        elif returns in ["reaction_energy", "activation_energy"]:
            feats, reaction_feats = self.model(mol_graphs, rxn_graphs, feats, metadata)
            preds = self.model.decode(feats, reaction_feats, metadata)

            state_dict = self.hparams.label_scaler[returns].state_dict()
            mean = state_dict["mean"]
            std = state_dict["std"]
            preds = preds[returns] * std + mean

            return preds

        else:
            supported = [
                "reaction_feature",
                "diff_feature_before_rxn_conv",
                "diff_feature_after_rxn_conv",
                "reaction_energy",
                "activation_energy",
            ]
            raise ValueError(
                f"Expect `returns` to be one of {supported}; got `{returns}`."
            )

    def training_step(self, batch, batch_idx):
        loss, preds, labels, indices = self.shared_step(batch, "train")
        self.update_metrics(preds, labels, "train")

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        self.compute_metrics("train")

    def validation_step(self, batch, batch_idx):
        loss, preds, labels, indices = self.shared_step(batch, "val")
        self.update_metrics(preds, labels, "val")

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        score = self.compute_metrics("val")

        # val/score used for early stopping and learning rate scheduler
        self.log(f"val/score", score, on_step=False, on_epoch=True, prog_bar=True)

        # time it
        delta_t, cumulative_t = self.timer.update()
        self.log("epoch time", delta_t, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cumulative time", cumulative_t, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, preds, labels, indices = self.shared_step(batch, "test")
        self.update_metrics(preds, labels, "test")

        return {"loss": loss}

    def test_epoch_end(self, outputs):
        self.compute_metrics("test")

    def shared_step(self, batch, mode):

        # ========== compute predictions ==========
        indices, mol_graphs, rxn_graphs, labels, metadata = batch

        # lightning cannot move dgl graphs to gpu, so do it manually
        mol_graphs = mol_graphs.to(self.device)
        rxn_graphs = rxn_graphs.to(self.device)

        nodes = ["atom", "global"]
        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}
        feats["bond"] = mol_graphs.edges["bond"].data.pop("feat")

        feats, reaction_feats = self.model(mol_graphs, rxn_graphs, feats, metadata)
        preds = self.model.decode(feats, reaction_feats, metadata)

        # ========== compute losses ==========
        all_loss = self.compute_loss(preds, labels)

        # ========== log the loss ==========
        total_loss = sum(all_loss.values())

        self.log_dict(
            {f"{mode}/loss/{task}": loss for task, loss in all_loss.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.log(
            f"{mode}/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        return total_loss, preds, labels, indices

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_scheduler == "reduce_on_plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.4, patience=50, verbose=True
            )
        elif self.hparams.lr_scheduler == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.epochs, eta_min=self.hparams.lr_min
            )
            if self.hparams.lr_warmup_step:
                scheduler = GradualWarmupScheduler(
                    optimizer,
                    multiplier=1,
                    total_epoch=self.hparams.lr_warmup_step,
                    after_scheduler=scheduler,
                )
        else:
            raise ValueError(f"Not supported lr scheduler: {self.hparams.lr_scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/score",
        }

    def init_metrics(self):
        metrics = nn.ModuleDict()

        for mode in ["metric_train", "metric_val", "metric_test"]:

            metrics[mode] = nn.ModuleDict()

            for task_name, task_setting in self.classification_tasks.items():
                n = task_setting["num_classes"]
                metrics[mode][task_name] = nn.ModuleDict(
                    {
                        "accuracy": pl.metrics.Accuracy(compute_on_step=False),
                        "precision": pl.metrics.Precision(
                            num_classes=n,
                            average="micro",
                            compute_on_step=False,
                        ),
                        "recall": pl.metrics.Recall(
                            num_classes=n,
                            average="micro",
                            compute_on_step=False,
                        ),
                        "f1": pl.metrics.F1(
                            num_classes=n,
                            average="micro",
                            compute_on_step=False,
                        ),
                    }
                )

            for task_name in self.regression_tasks:
                metrics[mode][task_name] = nn.ModuleDict(
                    {"mae": pl.metrics.MeanAbsoluteError(compute_on_step=False)}
                )

        return metrics

    def update_metrics(self, preds, labels, mode):
        """
        Update metric values at each step.
        """
        mode = "metric_" + mode

        for task_name in list(self.classification_tasks.keys()) + list(
            self.regression_tasks.keys()
        ):
            for metric in self.metrics[mode][task_name]:
                metric_obj = self.metrics[mode][task_name][metric]
                metric_obj(preds[task_name], labels[task_name])

    def compute_metrics(self, mode):
        """
        Compute metric and log it at each epoch.
        """
        mode = "metric_" + mode

        score = 0

        for task_name, task_setting in self.classification_tasks.items():
            for metric in self.metrics[mode][task_name]:
                metric_obj = self.metrics[mode][task_name][metric]
                out = metric_obj.compute()

                self.log(
                    f"{mode}/{metric}/{task_name}",
                    out,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

                metric_obj.reset()

                if metric in task_setting["to_score"]:
                    sign = task_setting["to_score"][metric]
                    score += out * sign

        for task_name, task_setting in self.regression_tasks.items():
            for metric in self.metrics[mode][task_name]:
                metric_obj = self.metrics[mode][task_name][metric]
                out = metric_obj.compute()

                # scale labels
                label_scaler = task_setting["label_scaler"]
                state_dict = self.hparams.label_scaler[label_scaler].state_dict()
                out *= state_dict["std"].to(self.device)

                self.log(
                    f"{mode}/{metric}/{task_name}",
                    out,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

                metric_obj.reset()

                if metric in task_setting["to_score"]:
                    sign = task_setting["to_score"][metric]
                    score += out * sign

        return score

    def create_model(self, params):
        """
        Create the model.

        Args:
            params:

        Return:
            The model.

        Example:

        model = ReactionRepresentation(
            in_feats=params.feature_size,
            embedding_size=params.embedding_size,
            # encoder
            molecule_conv_layer_sizes=params.molecule_conv_layer_sizes,
            molecule_num_fc_layers=params.molecule_num_fc_layers,
            molecule_batch_norm=params.molecule_batch_norm,
            molecule_activation=params.molecule_activation,
            molecule_residual=params.molecule_residual,
            molecule_dropout=params.molecule_dropout,
            reaction_conv_layer_sizes=params.reaction_conv_layer_sizes,
            reaction_num_fc_layers=params.reaction_num_fc_layers,
            reaction_batch_norm=params.reaction_batch_norm,
            reaction_activation=params.reaction_activation,
            reaction_residual=params.reaction_residual,
            reaction_dropout=params.reaction_dropout,
            # compressing
            compressing_layer_sizes=params.compressing_layer_sizes,
            compressing_layer_activation=params.compressing_layer_activation,
            # pooling method
            pooling_method=params.pooling_method,
            pooling_kwargs=params.pooling_kwargs,
        )

        return model
        """
        raise NotImplementedError

    def init_tasks(self):
        """
        Define the tasks (decoders) and the associated metrics.

        Example:

        self.classification_tasks = {
            # "bond_hop_dist": {
            #     "num_classes": self.hparams.bond_hop_dist_num_classes,
            #     "to_score": {"f1": 1},
            # }
        }

        self.regression_tasks = {
            # "reaction_energy": {
            #     "label_scaler": "reaction_energy",
            #     "to_score": {"mae": -1},
            # }
        }

        self.contrastive_tasks = {}

        """

        raise NotImplementedError

    def compute_loss(
        self, preds: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the loss for each task.

        Args:
            preds: {task_name, prediction} prediction for each task
            labels: {task_name, label} labels for each task

        Example:

        all_loss = {}

        # bond hop distance loss
        task = "bond_hop_dist"
        if task in self.classification_tasks:
            loss = F.cross_entropy(
                preds[task],
                labels[task],
                reduction="mean",
                weight=self.hparams.bond_hop_dist_class_weight.to(self.device),
            )
            all_loss[task] = loss

        Returns:
            {task_name, loss}
        """
        raise NotImplementedError

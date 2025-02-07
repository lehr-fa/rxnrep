"""
Base Lightning models.
"""

from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import lr_scheduler

from rxnrep.model.utils import TimeMeter


class BaseModel(pl.LightningModule):
    """
    Base Model for classification, regression, pretraining, and finetuning.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.backbone = self.init_backbone(self.hparams)

        decoder = self.init_decoder(self.hparams)
        for name in decoder:
            assert "decoder" in name, (
                f"Please add `decoder` to decoder name for `{name}` in "
                f"`self.init_decoder()`. We use the word `decoder` to detect which "
                "modules to freeze when doing fine tuning!"
            )
        self.decoder = nn.ModuleDict(decoder)

        self.regression_tasks = self.init_regression_tasks(self.hparams)
        self.classification_tasks = self.init_classification_tasks(self.hparams)
        self.metrics = self.init_metrics()

        self.timer = TimeMeter()

    def forward(
        self, mol_graphs, rxn_graphs, feats, metadata, return_mode: Optional[str] = None
    ):
        """
        Args:
            return_mode: select what to return. See below.

        Returns:
            If `None`, directly return the value returned by the self.model forward
            method. This is typically the features and reaction features returned
            by the decoder, e.g. (feats, reaction_feats). feats is a dictionary of
            atom, bond, and global features and reaction_feats is the reaction
            feature tensor.
            Values from the decoder can also be returned; currently supported ones are
            `reaction_energy`, `activation_energy`, and `reaction_type`.
        """

        if return_mode is None:
            return self.backbone(mol_graphs, rxn_graphs, feats, metadata)

        elif return_mode == "difference_feature":
            return self.get_difference_feature(mol_graphs, rxn_graphs, feats, metadata)

        elif return_mode == "pool_attention_score":
            return self.get_pool_attention_score(
                mol_graphs, rxn_graphs, feats, metadata
            )

        elif return_mode == "reaction_type":  # classification
            feats, reaction_feats = self.backbone(
                mol_graphs, rxn_graphs, feats, metadata
            )
            preds = self.decode(feats, reaction_feats, metadata)
            return preds[return_mode]

        else:
            supported = [
                None,
                "difference_feature",
                "poll_attention_score",
                "reaction_type",
            ]
            raise ValueError(
                f"Expect `return_mode` to be one of {supported}; got `{return_mode}`."
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
        if score is not None:
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
        if rxn_graphs is not None:
            rxn_graphs = rxn_graphs.to(self.device)

        nodes = ["atom", "global"]
        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}
        feats["bond"] = mol_graphs.edges["bond"].data.pop("feat")

        feats, reaction_feats = self(mol_graphs, rxn_graphs, feats, metadata)

        preds = self.decode(feats, reaction_feats, metadata)

        # ========== compute losses ==========
        all_loss = self.compute_loss(preds, labels)

        # ========== logger the loss ==========
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

        # learning rate scheduler
        scheduler = self._config_lr_scheduler(optimizer)

        if scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val/score",
            }

    def _config_lr_scheduler(self, optimizer):

        scheduler_name = self.hparams.lr_scheduler["scheduler_name"].lower()

        if scheduler_name == "reduce_on_plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.4, patience=50, verbose=True
            )
        elif scheduler_name == "cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.lr_scheduler["lr_warmup_step"],
                max_epochs=self.hparams.lr_scheduler["epochs"],
                eta_min=self.hparams.lr_scheduler["lr_min"],
            )
        elif scheduler_name == "none":
            scheduler = None
        else:
            raise ValueError(f"Not supported lr scheduler: {self.hparams.lr_scheduler}")

        return scheduler

    def init_metrics(self):
        metrics = nn.ModuleDict()

        for mode in ["metric_train", "metric_val", "metric_test"]:

            metrics[mode] = nn.ModuleDict()

            for task_name, task_setting in self.classification_tasks.items():
                n = task_setting["num_classes"]
                average = task_setting["average"]

                # binary or micro, num_classes not needed
                if n == 2 or average == "micro":
                    n = None

                metrics[mode][task_name] = nn.ModuleDict(
                    {
                        "accuracy": tm.Accuracy(
                            num_classes=n, average=average, compute_on_step=False
                        ),
                        "precision": tm.Precision(
                            num_classes=n, average=average, compute_on_step=False
                        ),
                        "recall": tm.Recall(
                            num_classes=n, average=average, compute_on_step=False
                        ),
                        "f1": tm.F1(
                            num_classes=n, average=average, compute_on_step=False
                        ),
                    }
                )

            for task_name in self.regression_tasks:
                metrics[mode][task_name] = nn.ModuleDict(
                    {"mae": tm.MeanAbsoluteError(compute_on_step=False)}
                )

        return metrics

    def update_metrics(self, preds, labels, mode):
        """
        Update metric values at each step.
        """
        mode = "metric_" + mode

        # regression metrics
        for task_name in self.regression_tasks:
            for metric in self.metrics[mode][task_name]:
                metric_obj = self.metrics[mode][task_name][metric]
                metric_obj(preds[task_name], labels[task_name])

        # classification metrics
        for task_name in self.classification_tasks:
            for metric in self.metrics[mode][task_name]:
                metric_obj = self.metrics[mode][task_name][metric]
                p = preds[task_name]
                if p.shape[1] == 1:
                    # binary
                    prop = torch.sigmoid(p.reshape(-1))
                else:
                    # multiclass
                    # prop = torch.softmax(p, dim=1)
                    prop = torch.argmax(p, dim=1)
                metric_obj(prop, labels[task_name])

    def compute_metrics(self, mode):
        """
        Compute metric and logger it at each epoch.
        """
        mode = "metric_" + mode

        # Do not set to 0 there in order to return None if there is no metric
        # contributes to score
        score = None

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
                    score = 0 if score is None else score
                    sign = task_setting["to_score"][metric]
                    score += out * sign

        for task_name, task_setting in self.regression_tasks.items():
            for metric in self.metrics[mode][task_name]:
                metric_obj = self.metrics[mode][task_name][metric]
                out = metric_obj.compute()

                # scale labels
                lb_scaler_name = task_setting["label_scaler"]
                label_scaler = self.hparams.dataset_info["label_scaler"][lb_scaler_name]
                state_dict = label_scaler.state_dict()
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
                    score = 0 if score is None else score
                    sign = task_setting["to_score"][metric]
                    score += out * sign

        return score

    def init_backbone(self, params) -> nn.Module:
        """
        Create backbone model.

        Return:
            A pytorch or lightning model that can be called like:
                `model(mol_graph, reaction_graph, feats, metadata)`
                and the model should return (mol_feats, reaction_feats).

        Returns:
            a model that is used as backbone
        """
        raise NotImplementedError

    def init_decoder(self, params) -> Dict[str, nn.Module]:
        """
        Create the decoder(s).

        The decoder(s) takes the output of backbone model as input and maps them into
        property, e.g. class label.

        Returns:
           A dictionary of decoders, {name: decoder}.
        """
        raise NotImplementedError

    def init_regression_tasks(self, params) -> Dict:
        """
        Define the the regression tasks used for computing metrics.

        Currently, `mae` metric is supported.

        Example:

        regression_tasks = {
            "reaction_energy": {
                "label_scaler": "reaction_energy",
                "to_score": {"mae": -1},
            },
            "activation_energy": {
                "label_scaler": "activation_energy",
                "to_score": {"mae": -1},
            }
        }

        return regression_tasks
        """
        return {}

    def init_classification_tasks(self, params) -> Dict:
        """
        Define the classification tasks used for computing metrics.

        Currently, `accuracy`, `recall`, `precision`, and f1 metrics are supported.

        Example:

        classification_tasks = {
            "reaction_type": {
                "num_classes": params.num_reaction_classes,
                "to_score": {"f1": 1},
                "average": "micro"
            }
        }

        return classification_tasks
        """
        return {}

    def decode(
        self,
        feats: Dict[str, torch.Tensor],
        reaction_feats: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        User the decoders to compute predictions from molecule and reaction features.

        Args:
            feats: atom and bond (maybe global) features of molecules
            reaction_feats: reaction features
            metadata:

        Returns:
            predictions: {decoder_name: value} predictions of the decoders.
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

    def get_pool_attention_score(self, mol_graphs, rxn_graphs, feats, metadata):
        """
        Returns:
            Dict of attention score

        """
        attn_score = self.backbone.get_pool_attention_score(
            mol_graphs, rxn_graphs, feats, metadata
        )
        return attn_score

    def get_difference_feature(self, mol_graphs, rxn_graphs, feats, metadata):
        """
        Returns:
            2D tensor of shape (B, D), where B is batch size and D is feature dim.
            Each row for a reaction.
        """
        diff_feat = self.backbone.get_difference_feature(
            mol_graphs, rxn_graphs, feats, metadata
        )
        return diff_feat


class BaseContrastiveModel(BaseModel):
    def compute_z(self, mol_graphs, rxn_graphs, metadata):
        nodes = ["atom", "global"]
        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}
        feats["bond"] = mol_graphs.edges["bond"].data.pop("feat")

        feats, reaction_feats = self(mol_graphs, rxn_graphs, feats, metadata)
        z = self.decode(feats, reaction_feats, metadata)

        return z

    def shared_step(self, batch, mode):

        # ========== compute predictions ==========
        (
            indices,
            (mol_graphs1, mol_graphs2),
            rxn_graphs,
            labels,
            (metadata1, metadata2),
        ) = batch

        # lightning cannot move dgl graphs to gpu, so do it manually
        mol_graphs1 = mol_graphs1.to(self.device)
        mol_graphs2 = mol_graphs2.to(self.device)
        if rxn_graphs is not None:
            rxn_graphs = rxn_graphs.to(self.device)

        z1 = self.compute_z(mol_graphs1, rxn_graphs, metadata1)
        z2 = self.compute_z(mol_graphs2, rxn_graphs, metadata2)
        preds = {"z1": z1, "z2": z2}

        # ========== compute losses ==========
        all_loss = self.compute_loss(preds, labels)

        # ========== logger the loss ==========
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


class BaseFinetuneModel(BaseModel):
    def configure_optimizers(self):
        """
        Different learning rate for prediction head and backbone encoder (if it is
        requested to be optimized).

        """

        # params for prediction head decoder
        assert (
            len(self.decoder) == 1
        ), f"Expect 1 decoder for finetune model, got {len(self.decoder)}"

        prediction_head = list(self.decoder.values())[0]

        params_group = [
            {
                "params": filter(
                    lambda p: p.requires_grad, prediction_head.parameters()
                ),
                "lr": self.hparams.lr,
                "weight_decay": self.hparams.weight_decay,
            }
        ]

        # params in encoder
        if self.hparams.finetune_tune_encoder:
            params_group.append(
                {
                    "params": filter(
                        lambda p: p.requires_grad, self.backbone.parameters()
                    ),
                    "lr": self.hparams.finetune_lr_encoder,
                    "weight_decay": self.hparams.weight_decay,
                }
            )

        optimizer = torch.optim.Adam(params_group)

        # learning rate scheduler
        scheduler = self._config_lr_scheduler(optimizer)

        if scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val/score",
            }

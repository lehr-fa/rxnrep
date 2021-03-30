"""
Base model for Constrative representation learning.
"""

import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import lr_scheduler

from rxnrep.scripts_contrastive.base_lit_model import BaseLightningModel as LitModel


class BaseLightningModel(LitModel):
    def configure_optimizers(self):
        """
        Different learning rate for prediction head and backbone encoder (if it is
        requested to be optimized).

        """

        # params for prediction head
        params_group = [
            {
                "params": filter(
                    lambda p: p.requires_grad, self.prediction_head.parameters()
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
                        lambda p: p.requires_grad, self.model.parameters()
                    ),
                    "lr": self.hparams.finetune_lr_encoder,
                    "weight_decay": self.hparams.weight_decay,
                }
            )

        optimizer = torch.optim.Adam(params_group)

        if self.hparams.lr_scheduler == "reduce_on_plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.4, patience=50, verbose=True
            )
        elif self.hparams.lr_scheduler == "cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.lr_warmup_step,
                max_epochs=self.hparams.epochs,
                eta_min=self.hparams.lr_min,
            )
        else:
            raise ValueError(f"Not supported lr scheduler: {self.hparams.lr_scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/score",
        }

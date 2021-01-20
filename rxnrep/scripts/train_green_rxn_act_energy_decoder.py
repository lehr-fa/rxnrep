import argparse
import warnings
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader

from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.data.green import GreenDataset
from rxnrep.model.model_rxn_act_energy_decoder import ReactionRepresentation
from rxnrep.scripts.launch_environment import PyTorchLaunch
from rxnrep.scripts.utils import (
    TimeMeter,
    get_latest_checkpoint_wandb,
    get_repo_git_commit,
    save_files_to_wandb,
)


class RxnRepLightningModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        # save params to be accessible via self.hparams
        self.save_hyperparameters(params)
        params = self.hparams

        self.model = ReactionRepresentation(
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
            # reaction energy decoder
            reaction_energy_decoder_hidden_layer_sizes=params.reaction_energy_decoder_hidden_layer_sizes,
            reaction_energy_decoder_activation=params.reaction_energy_decoder_activation,
            # reaction energy decoder
            activation_energy_decoder_hidden_layer_sizes=params.activation_energy_decoder_hidden_layer_sizes,
            activation_energy_decoder_activation=params.activation_energy_decoder_activation,
            # pooling method
            pooling_method=params.pooling_method,
            pooling_kwargs=params.pooling_kwargs,
        )

        # reaction cluster functions
        self.reaction_cluster_fn = {mode: None for mode in ["train", "val", "test"]}
        self.assignments = {mode: None for mode in ["train", "val", "test"]}
        self.centroids = {mode: None for mode in ["train", "val", "test"]}

        # metrics
        self.metrics = self._init_metrics()

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
        """
        nodes = ["atom", "bond", "global"]

        indices, mol_graphs, rxn_graphs, labels, metadata = batch

        # lightning cannot move dgl graphs to gpu, so do it manually
        mol_graphs = mol_graphs.to(self.device)
        rxn_graphs = rxn_graphs.to(self.device)

        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}

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

        else:
            supported = [
                "reaction_feature",
                "diff_feature_before_rxn_conv",
                "diff_feature_after_rxn_conv",
            ]
            raise ValueError(
                f"Expect `returns` to be one of {supported}; got `{returns}`."
            )

    def training_step(self, batch, batch_idx):
        loss, preds, labels, indices = self.shared_step(batch, "train")
        self._update_metrics(preds, labels, "train")

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        self._compute_metrics("train")

    def validation_step(self, batch, batch_idx):
        loss, preds, labels, indices = self.shared_step(batch, "val")
        self._update_metrics(preds, labels, "val")

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        # sum f1 to look for early stop and learning rate scheduler
        sum_f1 = self._compute_metrics("val")

        self.log(f"val/f1", sum_f1, on_step=False, on_epoch=True, prog_bar=True)

        # time it
        delta_t, cumulative_t = self.timer.update()
        self.log("epoch time", delta_t, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cumulative time", cumulative_t, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, preds, labels, indices = self.shared_step(batch, "test")
        self._update_metrics(preds, labels, "test")

        return {"loss": loss}

    def test_epoch_end(self, outputs):
        self._compute_metrics("test")

    def shared_step(self, batch, mode):

        # ========== compute predictions ==========
        indices, mol_graphs, rxn_graphs, labels, metadata = batch

        # lightning cannot move dgl graphs to gpu, so do it manually
        mol_graphs = mol_graphs.to(self.device)
        rxn_graphs = rxn_graphs.to(self.device)

        nodes = ["atom", "bond", "global"]
        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}

        feats, reaction_feats = self.model(mol_graphs, rxn_graphs, feats, metadata)
        preds = self.model.decode(feats, reaction_feats, metadata)

        # ========== compute losses ==========

        # loss for reaction energy
        preds["reaction_energy"] = preds["reaction_energy"].flatten()
        loss_reaction_energy = F.mse_loss(
            preds["reaction_energy"], labels["reaction_energy"]
        )

        preds["activation_energy"] = preds["activation_energy"].flatten()
        loss_activation_energy = F.mse_loss(
            preds["activation_energy"], labels["activation_energy"]
        )

        # total loss (maybe assign different weights)
        loss = loss_reaction_energy + loss_activation_energy

        # ========== log the loss ==========
        self.log_dict(
            {
                f"{mode}/loss/reaction_energy": loss_reaction_energy,
                f"{mode}/loss/activation_energy": loss_activation_energy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss, preds, labels, indices

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.4, patience=20, verbose=True
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/f1"}

    def _init_metrics(self):

        # (should be modules so that metric tensors can be placed in the correct device)
        metrics = nn.ModuleDict()
        for mode in ["metric_train", "metric_val", "metric_test"]:
            metrics[mode] = nn.ModuleDict(
                {
                    "reaction_energy": nn.ModuleDict(
                        {"mae": pl.metrics.MeanAbsoluteError(compute_on_step=False)}
                    ),
                    "activation_energy": nn.ModuleDict(
                        {"mae": pl.metrics.MeanAbsoluteError(compute_on_step=False)}
                    ),
                }
            )

        return metrics

    def _update_metrics(
        self,
        preds,
        labels,
        mode,
        keys=("reaction_energy", "activation_energy"),
    ):
        """
        update metric states at each step
        """
        mode = "metric_" + mode

        for key in keys:
            for mt in self.metrics[mode][key]:
                metric_obj = self.metrics[mode][key][mt]
                metric_obj(preds[key], labels[key])

    def _compute_metrics(
        self,
        mode,
        keys=("reaction_energy", "activation_energy"),
    ):
        """
        compute metric and log it at each epoch
        """
        mode = "metric_" + mode

        sum_f1 = 0
        for key in keys:
            for name in self.metrics[mode][key]:

                metric_obj = self.metrics[mode][key][name]
                value = metric_obj.compute()

                # reaction energy and activation energy are scaled, multiple std to get
                # back to the original
                if key in ["reaction_energy", "activation_energy"]:
                    value *= self.hparams.label_std[key].to(self.device)

                self.log(
                    f"{mode}/{name}/{key}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

                # reset is called automatically somewhere by lightning, here we call it
                # explicitly just in case
                metric_obj.reset()

                # NOTE, we abuse the sum_f1 to add the mae of energy (smaller the better)
                # prediction as well
                if name == "mae":
                    sum_f1 -= value

        return sum_f1


def parse_args():
    parser = argparse.ArgumentParser(description="Reaction Representation")

    # ========== dataset ==========
    prefix = "/Users/mjwen/Documents/Dataset/activation_energy_Green/"

    fname_tr = prefix + "wb97xd3_n200_processed_train.tsv"
    fname_val = fname_tr
    fname_test = fname_tr

    parser.add_argument("--trainset_filename", type=str, default=fname_tr)
    parser.add_argument("--valset_filename", type=str, default=fname_val)
    parser.add_argument("--testset_filename", type=str, default=fname_test)
    parser.add_argument(
        "--dataset_state_dict_filename", type=str, default="dataset_state_dict.yaml"
    )

    # ========== model ==========
    # embedding
    parser.add_argument("--embedding_size", type=int, default=24)

    # encoder
    parser.add_argument(
        "--molecule_conv_layer_sizes", type=int, nargs="+", default=[64, 64, 64]
    )
    parser.add_argument("--molecule_num_fc_layers", type=int, default=2)
    parser.add_argument("--molecule_batch_norm", type=int, default=1)
    parser.add_argument("--molecule_activation", type=str, default="ReLU")
    parser.add_argument("--molecule_residual", type=int, default=1)
    parser.add_argument("--molecule_dropout", type=float, default="0.0")
    parser.add_argument(
        "--reaction_conv_layer_sizes", type=int, nargs="+", default=[64, 64, 64]
    )
    parser.add_argument("--reaction_num_fc_layers", type=int, default=2)
    parser.add_argument("--reaction_batch_norm", type=int, default=1)
    parser.add_argument("--reaction_activation", type=str, default="ReLU")
    parser.add_argument("--reaction_residual", type=int, default=1)
    parser.add_argument("--reaction_dropout", type=float, default="0.0")

    # ========== pooling ==========
    parser.add_argument(
        "--pooling_method",
        type=str,
        default="set2set",
        help="set2set or hop_distance",
    )
    parser.add_argument("--max_hop_distance", type=int, default=3)

    # ========== decoder ==========

    # reaction energy decoder

    parser.add_argument(
        "--reaction_energy_decoder_hidden_layer_sizes",
        type=int,
        nargs="+",
        default=[64],
    )
    parser.add_argument(
        "--reaction_energy_decoder_activation", type=str, default="ReLU"
    )

    # parser.add_argument(
    #     "--activation_energy_decoder_hidden_layer_sizes",
    #     type=int,
    #     nargs="+",
    #     default=[64],
    # )
    # parser.add_argument(
    #     "--activation_energy_decoder_activation", type=str, default="ReLU"
    # )

    # ========== training ==========

    # restore
    parser.add_argument("--restore", type=int, default=0, help="restore training")

    # accelerator
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes")
    parser.add_argument(
        "--gpus", type=int, default=None, help="number of gpus per node"
    )
    parser.add_argument(
        "--accelerator", type=str, default=None, help="backend, e.g. `ddp`"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of workers for dataloader"
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=1,
        help="number of processes for constructing graphs in dataset",
    )

    # training algorithm
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")

    ####################
    # helper args
    ####################
    # encoder
    parser.add_argument(
        "--conv_layer_size",
        type=int,
        default=64,
        help="hidden layer size for mol and rxn conv",
    )
    parser.add_argument("--num_mol_conv_layers", type=int, default=2)
    parser.add_argument("--num_rxn_conv_layers", type=int, default=2)

    # decoder
    parser.add_argument("--num_rxn_energy_decoder_layers", type=int, default=2)

    ####################
    args = parser.parse_args()
    ####################

    ####################
    # adjust args
    ####################
    # encoder
    args.molecule_conv_layer_sizes = [args.conv_layer_size] * args.num_mol_conv_layers
    args.reaction_conv_layer_sizes = [args.conv_layer_size] * args.num_rxn_conv_layers
    if args.num_rxn_conv_layers == 0:
        args.reaction_dropout = 0

    # decoder
    val = 2 * args.conv_layer_size
    args.reaction_energy_decoder_hidden_layer_sizes = [
        max(val // 2 ** i, 50) for i in range(args.num_rxn_energy_decoder_layers)
    ]

    # pooling
    if args.pooling_method == "set2set":
        args.pooling_kwargs = None
    elif args.pooling_method == "hop_distance":
        args.pooling_kwargs = {"max_hop_distance": args.max_hop_distance}

    args.activation_energy_decoder_hidden_layer_sizes = (
        args.reaction_energy_decoder_hidden_layer_sizes
    )
    args.activation_energy_decoder_activation = args.reaction_energy_decoder_activation

    return args


def load_dataset(args):

    # check dataset state dict if restore model
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

    trainset = GreenDataset(
        filename=args.trainset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        max_hop_distance=args.max_hop_distance,
        init_state_dict=state_dict_filename,
        num_processes=args.nprocs,
    )

    state_dict = trainset.state_dict()

    valset = GreenDataset(
        filename=args.valset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        max_hop_distance=args.max_hop_distance,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
    )

    testset = GreenDataset(
        filename=args.testset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        max_hop_distance=args.max_hop_distance,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
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
    args.label_std = trainset.label_std

    args.feature_size = trainset.feature_size

    return train_loader, val_loader, test_loader


def main():
    print("\nStart training at:", datetime.now())

    pl.seed_everything(25)

    args = parse_args()

    # ========== dataset ==========
    train_loader, val_loader, test_loader = load_dataset(args)

    # ========== model ==========
    model = RxnRepLightningModel(args)

    # ========== trainer ==========

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/f1", mode="max", save_last=True, save_top_k=5, verbose=False
    )
    early_stop_callback = EarlyStopping(
        monitor="val/f1", min_delta=0.0, patience=50, mode="max", verbose=True
    )

    # logger
    log_save_dir = Path("wandb").resolve()
    project = "tmp-rxnrep"

    # restore model, epoch, shared_step, LR schedulers, apex, etc...
    if args.restore and log_save_dir.exists():
        # restore
        checkpoint_path = get_latest_checkpoint_wandb(log_save_dir, project)
    else:
        # create new
        checkpoint_path = None

    if not log_save_dir.exists():
        # put in try except in case it throws errors in distributed training
        try:
            log_save_dir.mkdir()
        except FileExistsError:
            pass
    wandb_logger = WandbLogger(save_dir=log_save_dir, project=project)

    # cluster environment to use torch.distributed.launch, e.g.
    # python -m torch.distributed.launch --use_env --nproc_per_node=2 <this_script.py>
    cluster = PyTorchLaunch()

    #
    # To run ddp on cpu, comment out `gpus` and `plugins`, and then set
    # `num_processes=2`, and `accelerator="ddp_cpu"`. Also note, for this script to
    # work, size of val (test) set should be larger than
    # `--num_centroids*num_processes`; otherwise clustering will raise an error,
    # but ddp_cpu cannot respond to it. As a result, it will stuck there.
    #

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        accelerator=args.accelerator,
        plugins=[cluster],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        resume_from_checkpoint=checkpoint_path,
        progress_bar_refresh_rate=100,
        flush_logs_every_n_steps=50,
        weights_summary="top",
        # profiler="simple",
        # deterministic=True,
    )

    # ========== fit and test ==========
    trainer.fit(model, train_loader, val_loader)
    trainer.test(test_dataloaders=test_loader)

    # ========== save files to wandb ==========
    # Do not do this before trainer, since this might result in the initialization of
    # multiple wandb object when training in distribution mode
    if (
        args.gpus is None
        or args.gpus == 1
        or (args.gpus > 1 and cluster.local_rank() == 0)
    ):
        save_files_to_wandb(wandb_logger, __file__, ["sweep.py", "submit.sh"])

    print("\nFinish training at:", datetime.now())


if __name__ == "__main__":

    repo_path = "/Users/mjwen/Applications/rxnrep"
    latest_commit = get_repo_git_commit(repo_path)
    print("Git commit:\n", latest_commit)

    main()
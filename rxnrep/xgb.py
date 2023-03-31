from functools import partial
import logging
from typing import Tuple

import numpy as np
import torch
import torchmetrics as tm
import xgboost as xgb
import hydra
import wandb
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

from rxnrep.data.transforms import Transform
from rxnrep.utils.config import get_datamodule_config
from rxnrep.utils.wandb import save_files_to_wandb, write_running_metadata

logger = logging.getLogger(__file__)


def xgb_F1Score(preds: np.ndarray, dtest: xgb.DMatrix, num_classes: int) -> float:
    """
    Custom XGBoost Metric for global F1 score.

    Args:
        preds (np.ndarray): Predictions [B].
        dtest (xgb.DMatrix): Test data (x: [B, D], y: [B]).

    Returns:
        float: Metric value.
    """

    y = dtest.get_label()  # [B]

    f1_metric = tm.F1(num_classes=num_classes, average='micro', compute_on_step=False)
    f1_metric(preds, y)

    return f1_metric.compute().item()


def xgb_fit(config: DictConfig, train_feats, train_labels, val_feats, val_labels, num_classes):
    
    params = {
        'booster': config.model.xgb.xgb.booster,
        'eta': config.model.xgb.xgb.learning_rate,
        'gamma': config.model.xgb.xgb.gamma,
        'max_depth': config.model.xgb.xgb.max_depth,
        'min_child_weight': config.model.xgb.xgb.min_child_weight,
        'colsample_bytree': config.model.xgb.xgb.colsample_bytree,
        'colsample_bylevel': config.model.xgb.xgb.colsample_bylevel,
        'subsample': config.model.xgb.xgb.xgb_subsampling_rate,
        'sampling_method': config.model.xgb.xgb.xgb_subsampling_mode,
        'scale_pos_weight': config.model.xgb.xgb.scale_pos_weight,
        'objective': config.model.xgb.xgb.objective,
        'num_class': num_classes,
        'seed': config.seed
    }
    evals_result = {}
    
    train_data = xgb.DMatrix(train_feats, label=train_labels)
    val_data = xgb.DMatrix(val_feats, label=val_labels)
    
    metric = partial(xgb_F1Score, num_classes=num_classes)
    
    xgb_model = xgb.train(params, train_data, evals=[(train_data, 'train'), (val_data, 'validation')], 
                          evals_result=evals_result, num_boost_round=config.model.xgb.xgb.num_round, feval=metric, maximize=True, 
                          early_stopping_rounds=config.model.xgb.xgb.num_early_stopping_round, verbose_eval=True)
    
    return xgb_model


def xgb_eval(xgb_model, test_feats, test_labels, num_classes):
    preds = xgb_model.predict(test_feats, iteration_range=(0, xgb_model.best_iteration+1), strict_shape=True)
    preds = np.argmax(preds, axis=1)
    
    acc_metric = tm.Accuracy(num_classes=num_classes, average='micro', compute_on_step=False)
    prec_metric = tm.Precision(num_classes=num_classes, average='micro', compute_on_step=False)
    recall_metric = tm.Recall(num_classes=num_classes, average='micro', compute_on_step=False)
    f1_metric = tm.F1(num_classes=num_classes, average='micro', compute_on_step=False)
    
    acc_metric(preds, test_labels)
    prec_metric(preds, test_labels)
    recall_metric(preds, test_labels)
    f1_metric(preds, test_labels)
    
    acc = acc_metric.compute().item()
    prec = prec_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()
    
    out = {'accuracy': acc,
           'precision': prec,
           'recall': recall,
           'f1': f1}
    
    return out



def inference(config: DictConfig) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                           Tuple[np.ndarray, np.ndarray], 
                                           Tuple[np.ndarray, np.ndarray],
                                           int]:
    """
    Performs inference of backbone model.

    Instantiates all PyTorch Lightning objects from config.

    Args:
        config: Configuration composed by Hydra.

    Returns:
        Features, labels for train, val, and test datasets; num_classes.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed)

    #
    #  Init datamodule
    #
    dm_config, _ = get_datamodule_config(config)

    logger.info(f"Instantiating datamodule: {dm_config._target_}")

    # contrastive
    if "transform1" in config and "transform2" in config:
        transform1: Transform = hydra.utils.instantiate(config.transform1)
        transform2: Transform = hydra.utils.instantiate(config.transform2)
        datamodule: LightningDataModule = hydra.utils.instantiate(
            dm_config, transform1=transform1, transform2=transform2
        )

    # predictive (regression/classification)
    else:
        datamodule: LightningDataModule = hydra.utils.instantiate(dm_config)

    # manually call them to get data needed for setting up the model
    # (Lightning still ensures the method runs on the correct devices)
    datamodule.prepare_data()
    datamodule.setup()
    logger.info(f"Finished instantiating datamodule: {dm_config._target_}")

    # datamodule info passed to model
    dataset_info = datamodule.get_to_model_info()

    #
    # Init Lightning model
    #

    # regular training
    if "decoder" in config.model:
        # encoder only provides args, decoder has the actual _target_
        logger.info(f"Instantiating model: {config.model.decoder.model_class._target_}")

        if "encoder" in config.model:
            encoder = config.model.encoder
            model: LightningModule = hydra.utils.instantiate(
                config.model.decoder.model_class,
                dataset_info=dataset_info,
                **encoder,
                **config.optimizer,
            )

        # when using morgan feats, there is no encoder
        else:
            model: LightningModule = hydra.utils.instantiate(
                config.model.decoder.model_class,
                dataset_info=dataset_info,
                **config.optimizer,
            )

    # finetune
    elif "finetuner" in config.model:
        logger.info(
            f"Instantiating model: {config.model.finetuner.model_class._target_}"
        )
        model: LightningModule = hydra.utils.instantiate(
            config.model.finetuner.model_class,
            dataset_info=dataset_info,
            **config.optimizer,
        )

    else:
        raise ValueError("Only support `decoder` model and `finetune` model")

    #
    # Inference
    #
    
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()
    test_dl = datamodule.test_dataloader()
    
    logger.info("Starting feature computation!")
    
    train_feats, train_labels = inference_on_dl(model, train_dl)
    val_feats, val_labels = inference_on_dl(model, val_dl)
    test_feats, test_labels = inference_on_dl(model, test_dl)
        
    #in_size=model.backbone.backbone.reaction_feats_size
    

    return (train_feats, train_labels), (val_feats, val_labels), (test_feats, test_labels), dataset_info["num_reaction_classes"]


def inference_on_dl(model, dl):
    feats_out = []
    labels_out = []
    
    for batch in dl:
        indices, mol_graphs, rxn_graphs, labels, metadata = batch

        # lightning cannot move dgl graphs to gpu, so do it manually
        mol_graphs = mol_graphs.to(model.device)
        if rxn_graphs is not None:
            rxn_graphs = rxn_graphs.to(model.device)

        nodes = ["atom", "global"]
        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}
        feats["bond"] = mol_graphs.edges["bond"].data.pop("feat")

        with torch.no_grad():
            feats, reaction_feats = model(mol_graphs, rxn_graphs, feats, metadata, return_mode=None)
        
        feats_out.append(reaction_feats.cpu().numpy())
        labels_out.append(labels.cpu().numpy())
    
    return np.concatenate(feats_out), np.concatenate(labels_out)



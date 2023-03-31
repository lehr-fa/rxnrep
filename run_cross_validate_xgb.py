"""
Cross validation.

Data split:
- the dataset to split for cv should be provided as dm.trainset_filename
- the dataset will be split into k folds, each with a train.tsv and test.tsv file,
  organized like:
    - cv_fold_1
        - train.tsv
        - test.tsv
    - cv_fold_2
        - train.tsv
        - test.tsv
    - cv_fold_3
        - train.tsv
        - test.tsv

- for each fold, the train.tsv is reassigned to dm.trainset_filename
- for each fold, test.tsv will be reassigned to dm.valset_filename if it is not provided,
  and will be reassigned to dm.test_filename if it is not provided
"""
import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from rxnrep.xgb import inference, xgb_fit, xgb_eval
from rxnrep.utils.config import (
    dump_config,
    get_datamodule_config,
    get_wandb_logger_config,
    merge_configs,
    print_config,
)
from rxnrep.utils.cross_validate import compute_metric_statistics
from rxnrep.utils.io import to_path
from rxnrep.utils.wandb import copy_pretrained_model

logger = logging.getLogger(__file__)

# HYDRA_FULL_ERROR=1 for complete stack trace
os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(config_path="configs", config_name="config_cross_validate_xgb.yaml")
def main(cfg: DictConfig):

    # The copy_trained_model fn here is only for test purpose, should remove
    if "finetuner" in cfg.model:
        wandb_id = cfg.get("pretrained_wandb_id", None)
        if wandb_id:
            path = to_path(cfg.original_working_dir).joinpath("outputs")
            copy_pretrained_model(wandb_id, source_dir=path)

    # Update cfg, new or modified ones by model encoder and decoder
    # won't change the model behavior, only add some helper args
    if "finetuner" in cfg.model:
        cfg_update = hydra.utils.call(cfg.model.finetuner.cfg_adjuster, cfg)
    else:
        cfg_update = hydra.utils.call(cfg.model.decoder.cfg_adjuster, cfg)

    # Reset CV filename to trainset filename of datamodule
    dm_cfg, _ = get_datamodule_config(cfg)
    cv_filename = OmegaConf.create(
        {"cross_validate": {"filename": dm_cfg.trainset_filename}}
    )
    cfg_update = merge_configs(cfg_update, cv_filename)

    # Merge cfg
    cfg_final = merge_configs(cfg, cfg_update)
    OmegaConf.set_struct(cfg_final, True)

    # Save configs to file
    dump_config(cfg, "hydra_cfg_original.yaml")
    dump_config(cfg_update, "hydra_cfg_update.yaml")
    dump_config(cfg_final, "hydra_cfg_final.yaml")

    # It does not bother to print it again, useful for debug
    print_config(cfg_final, label="CONFIG", resolve=True, sort_keys=True)

    # Get CV data split

    data_splits = hydra.utils.call(cfg_final.cross_validate)

    # Determine whether valset_filename and testset_filename are provided in datamodule
    # (should not do this in the loop since dm_cfg.testset_filename is reset)
    dm_cfg, _ = get_datamodule_config(cfg_final)

    if dm_cfg.valset_filename:
        dm_has_valset = True
    else:
        dm_has_valset = False

    if dm_cfg.testset_filename:
        dm_has_testset = True
    else:
        dm_has_testset = False

    metric_scores = []
    for i, (trainset, testset) in enumerate(data_splits):

        OmegaConf.set_struct(cfg_final, False)

        # Update datamodule (trainset_filename, valset_filename, and testset_filename)
        dm_cfg.trainset_filename = str(trainset)
        if not dm_has_valset:
            dm_cfg.valset_filename = str(testset)
        if not dm_has_testset:
            dm_cfg.testset_filename = str(testset)

        # Update wandb logger info (save_dir)
        wandb_logger_cfg = get_wandb_logger_config(cfg_final)
        wandb_save_dir = Path.cwd().joinpath(f"cv_fold_{i}")
        if not wandb_save_dir.exists():
            wandb_save_dir.mkdir()
        wandb_logger_cfg.save_dir = wandb_save_dir.as_posix()

        OmegaConf.set_struct(cfg_final, True)

        logger.info(
            f"Cross validation fold {i}. Set wandb save_dir to: {wandb_save_dir}"
        )
        logger.info(
            f"Cross validation fold {i}. With trainset: {dm_cfg.trainset_filename}, "
            f"valset: {dm_cfg.valset_filename}, and testset: {dm_cfg.testset_filename}"
        )
        
        (train_feats, train_labels), (val_feats, val_labels), (test_feats, test_labels), num_classes = inference(cfg_final)
        
        xgb_model = xgb_fit(cfg_final, train_feats, train_labels, val_feats, val_labels, num_classes)
        
        score = xgb_eval(xgb_model, test_feats, test_labels, num_classes)
        
        metric_scores.append(score)

    metrics, mean, std = compute_metric_statistics(metric_scores)
    logger.info(f"Cross validation metrics (all): {metrics}")
    logger.info(f"Cross validation metrics (mean): {mean}")
    logger.info(f"Cross validation metrics (std): {std}")


if __name__ == "__main__":
    main()

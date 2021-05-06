import os

import hydra
from omegaconf import DictConfig, OmegaConf

from rxnrep.train import train
from rxnrep.utils.config import dump_config, get_restore_config, print_config

# HYDRA_FULL_ERROR=1 for complete stack trace
os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):

    # Update cfg, new or modified ones by encoder and decoder
    # won't change the model behavior, only add some helper args
    cfg_update = hydra.utils.call(cfg.model.decoder.cfg_adjuster, cfg)

    # Restore cfg from latest run
    if cfg.restore:
        cfg_restore = get_restore_config(cfg)
        cfg_update = OmegaConf.merge(cfg_update, cfg_restore)

    # Merge cfg
    OmegaConf.set_struct(cfg, False)
    cfg_final = OmegaConf.merge(cfg, cfg_update)
    OmegaConf.set_struct(cfg_final, True)

    # Save configs to file
    dump_config(cfg, "hydra_cfg_original.yaml")
    dump_config(cfg_update, "hydra_cfg_update.yaml")
    dump_config(cfg_final, "hydra_cfg_final.yaml")

    # It does not bother to print it again, useful for debug
    print_config(cfg_final, label="CONFIG", resolve=True, sort_keys=True)

    # train the model
    train(cfg_final)


if __name__ == "__main__":
    main()

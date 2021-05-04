import glob
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from rxnrep.utils.io import create_directory, to_path, yaml_dump

logger = logging.getLogger(__name__)


def get_git_repo_commit(repo_path: Path) -> str:
    """
    Get the latest git commit info of a github repository.

    Args:
        repo_path: path to the repo

    Returns:
        latest commit info
    """
    output = subprocess.check_output(["git", "log"], cwd=to_path(repo_path))
    output = output.decode("utf-8").split("\n")[:6]
    latest_commit = "\n".join(output)

    return latest_commit


def get_hostname() -> str:
    """
    Get the hostname of the machine.

    Returns:
        hostname
    """
    output = subprocess.check_output("hostname")
    hostname = output.decode("utf-8").strip()

    return hostname


def write_running_metadata(
    git_repo: Optional[Path] = None, filename: str = "running_metadata.yaml"
):
    """
    Write additional running metadata to a file and then copy it to wandb.

    Currently, we write:
    - the running dir, i.e. cwd
    - git repo commit, optional

    Args:
        filename: name of the file to write
        git_repo: path to the git repo, if None, do not use this info.
    """

    d = {"running_dir": Path.cwd().as_posix(), "hostname": get_hostname()}
    if git_repo is not None:
        d["git_commit"] = get_git_repo_commit(git_repo)

    yaml_dump(d, filename)


@rank_zero_only
def save_files_to_wandb(wandb_logger, files: List[str] = None):
    """
    Save files to wandb. The files should be given relative to cwd.

    Args:
        wandb_logger: lightning wandb logger
        files: name of the files in the running directory to save. If a file does not
            exist, it is silently ignored.
    """
    wandb = wandb_logger.experiment

    for f in files:
        p = Path.cwd().joinpath(f)
        if p.exists():
            wandb.save(p.as_posix(), policy="now", base_path=".")


def get_hydra_latest_run(path: Union[str, Path], index: int = -1) -> Union[Path, None]:
    """
    Find the latest hydra running directory in the hydra `outputs`.

    This assumes the hydra outputs look like:

    - outputs
      - 2021-05-02
        - 11-26-08
        - 12-01-19
      - 2021-05-03
        - 09-08-01
        - 10-10-01

    Args:
        path: path to the `outputs` directory. For example, this should be `../../`
        relative to the hydra current working directory cwd.
        index: index to the hydra runs to return. By default, -1 returns the last one.
            But this may not be the one we want when we are calling this from a hydra
            run, in that case, -1 will index to itself. In this case, we can pass -2 to
            get the latest one before the current one.

    Returns:
        Path to the latest hydra run. `None` if not such path can be found.
    """

    path = to_path(path)
    all_paths = []
    for data in os.listdir(path):
        for time in os.listdir(path.joinpath(data)):
            all_paths.append(path.joinpath(data, time))
    all_paths = sorted(all_paths, key=lambda p: p.as_posix())

    if len(all_paths) < abs(index):
        return None
    else:
        return all_paths[index]


def get_dataset_state_dict_latest_run(
    path: Union[str, Path], name: str
) -> Union[str, None]:
    """
    Get path to the dataset state dict of the latest run.

    Args:
        path: path to hydra `outputs` directory. For example, this should be `../../`
            relative to the hydra current working directory cwd.
        name: name of the dataset state dict file

    Returns:
        path to the dataset state dict yaml file. None if cannot find the file
    """
    latest = get_hydra_latest_run(path, index=-2)

    if latest is not None:
        dataset_state_dict = latest.joinpath(name)
        if dataset_state_dict.exists():
            return dataset_state_dict.as_posix()

    return None


def get_wandb_identifier_latest_run(path: Union[str, Path]) -> Union[str, None]:
    """
    Get the wandb unique identifier of the latest run.

    Args:
        path: path to hydra `outputs` directory. For example, this should be `../../`
            relative to the hydra current working directory cwd.

    Returns:
        identifier str, None if cannot find the run
    """
    latest = get_hydra_latest_run(path, index=-2)

    if latest is not None:
        latest_run = latest.joinpath("wandb", "latest-run").resolve()
        if latest_run.exists():
            identifier = str(latest_run).split("-")[-1]
            if identifier != "run":
                return identifier

    return None


def get_wandb_checkpoint_latest_run(
    path: Union[str, Path], project: str
) -> Union[str, None]:
    """
    Get the wandb checkpoint of the latest run.

    Args:
        path: path to hydra `outputs` directory. For example, this should be `../../`
            relative to the hydra current working directory cwd.
        project: project name of the wandb run

    Returns:
        path to the latest checkpoint, None if it does not exist
    """
    latest = get_hydra_latest_run(path, index=-2)

    if latest is not None:
        identifier = get_wandb_identifier_latest_run(path)
        if identifier is not None:
            checkpoint = latest.joinpath(
                project, identifier, "checkpoints", "last.ckpt"
            ).resolve()
            if checkpoint.exists():
                return checkpoint.as_posix()

    return None


def get_restore_config(config: DictConfig) -> DictConfig:
    """
    Get the config info used to restore the model from the latest run.

    This includes: dataset state dict path, checkpoint path, and wandb identifier.

    Args:
        config: hydra config

    Returns:
        DictConfig with info related to restoring the model.
    """

    # Get datamodule config
    # we do the for loop to get it because we group datamodule config into: predictive,
    # contrastive, finetune...
    for name in config.datamodule:
        dm_name = name
        dm_config = config.datamodule[name]
        break

    dataset_state_dict_filename = dm_config.get(
        "state_dict_filename", "dataset_state_dict.yaml"
    )
    project = config.logger.wandb.project
    path = to_path(config.original_working_dir).joinpath("outputs")

    dataset_state_dict = get_dataset_state_dict_latest_run(
        path, dataset_state_dict_filename
    )
    checkpoint = get_wandb_checkpoint_latest_run(path, project)
    identifier = get_wandb_identifier_latest_run(path)

    d = {
        "datamodule": {dm_name: {"restore_state_dict_filename": dataset_state_dict}},
        "callbacks": {"wandb": {"id": identifier}},
        "trainer": {"resume_from_checkpoint": checkpoint},
    }

    logger.info(f"Restoring training with automatically determined info: {d}")

    if dataset_state_dict is None:
        logger.warning(
            f"Trying to automatically restore dataset state dict, but cannot find latest "
            f"dataset state dict file. Now, we set it to `None` to compute dataset "
            f"statistics (e.g. feature mean and standard deviation) from the trainset."
        )
    if checkpoint is None:
        logger.warning(
            f"Trying to automatically restore model from checkpoint, but cannot find "
            f"latest checkpoint file. Proceed without restoring."
        )
    if identifier is None:
        logger.warning(
            f"Trying to automatically restore training with the same wandb identifier, "
            f"but cannot find the identifier of latest run. A new wandb identifier will "
            f"be assigned."
        )

    restore_config = OmegaConf.create(d)

    return restore_config


def get_wandb_run_path(identifier: str, path="."):
    """
    Args:
        identifier: wandb unique identifier of experiment, e.g. 2i3rocdl
        path: root path to search
    Returns:
        path to the wandb run directory:
        e.g. running_dir/job_0/wandb/wandb/run-20201210_160100-3kypdqsw
    """
    for root, dirs, files in os.walk(path):
        if "wandb" not in root:
            continue
        for d in dirs:
            if d.startswith("run-") or d.startswith("offline-run-"):
                if d.split("-")[-1] == identifier:
                    return os.path.abspath(os.path.join(root, d))

    raise RuntimeError(f"Cannot found job {identifier} in {path}")


def get_wandb_checkpoint_path(identifier: str, path="."):
    """
    Args:
        identifier: wandb unique identifier of experiment, e.g. 2i3rocdl
        path: root path to search
    Returns:
        path to the wandb checkpoint directory:
        e.g. running_dir/job_0/wandb/<project_name>/<identifier>/checkpoints
    """
    for root, dirs, files in os.walk(path):
        if root.endswith(f"{identifier}/checkpoints"):
            return os.path.abspath(root)

    raise RuntimeError(f"Cannot found job {identifier} in {path}")


def copy_trained_model(
    identifier: str, source_dir: Path = ".", target_dir: Path = "trained_model"
):
    """
    Copy the last checkpoint and dataset_state_dict.yaml to a directory.

    Args:
        identifier: wandb unique identifier of experiment, e.g. 2i3rocdl
        source_dir:
        target_dir:
    """
    # create target dir
    target_dir = to_path(target_dir)
    create_directory(target_dir, is_directory=True)

    # copy checkpoint file
    ckpt_dir = get_wandb_checkpoint_path(identifier, source_dir)
    print("Checkpoint path:", ckpt_dir)

    checkpoints = glob.glob(os.path.join(ckpt_dir, "epoch=*.ckpt"))
    checkpoints = sorted(checkpoints)
    shutil.copy(checkpoints[-1], target_dir.joinpath("checkpoint.ckpt"))

    # copy config.yaml file
    run_path = get_wandb_run_path(identifier, source_dir)
    print("wandb run path:", run_path)

    f = to_path(run_path).joinpath("files", "config.yaml")
    shutil.copy(f, target_dir.joinpath("config.yaml"))

    # copy dataset state dict
    f = to_path(run_path).joinpath("files", "dataset_state_dict.yaml")
    shutil.copy(f, target_dir.joinpath("dataset_state_dict.yaml"))


def load_checkpoint_tensorboard(save_dir="./lightning_logs"):
    """
    Get the latest checkpoint path of tensorboard logger.
    """
    path = Path(save_dir).resolve()
    versions = os.listdir(path)
    v = sorted(versions)[-1]
    checkpoints = os.listdir(path.joinpath(f"{v}/checkpoints"))
    ckpt = sorted(checkpoints)[-1]

    ckpt_path = str(path.joinpath(f"{v}/checkpoints/{ckpt}"))

    return ckpt_path
import mlflow
import numpy as np
import os
import sys

from typing import Dict, Any, Tuple, Union, Optional, List
from datetime import datetime
from stable_baselines3.common.logger import (
    HumanOutputFormat,
    KVWriter,
    Logger,
    configure,
    INFO,
)
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import git
from src import REPO_PATH
from loguru import logger


@dataclass
class MlflowConfig:
    tracking_uri: str
    """MLflow tracking URI"""

    experiment_name: str
    """MLflow experiment name"""

    run_name: Optional[str] = None
    """MLflow run name"""


class MLflowOutputFormat(KVWriter):
    """Dumps key/value pairs into MLflow's numeric format."""

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


class SilentLogger(Logger):
    def __init__(
        self,
        folder: Optional[str] = None,
        output_formats: Optional[List[KVWriter]] = None,
    ):
        self.name_to_value = defaultdict(
            float
        )  # Preserve the original Logger attributes
        self.name_to_count = defaultdict(int)
        self.name_to_excluded = defaultdict(
            lambda: None
        )  # Use a dictionary for exclusions
        self.level = INFO
        self.folder = folder
        self.output_formats = output_formats or []

        if folder is not None:
            os.makedirs(folder, exist_ok=True)


def create_mlflow_logger():
    logger = SilentLogger(
        folder=None,
        output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
    )
    return logger


def is_branch_exist(branch_name: str) -> bool:
    repo = git.Repo(REPO_PATH)
    # Check both local and remote branches
    all_branches = [ref.name for ref in repo.references]
    # Remove 'refs/heads/' prefix for local branches and 'refs/remotes/origin/' for remote branches
    branch_names = [
        ref.replace("refs/heads/", "").replace("refs/remotes/origin/", "")
        for ref in all_branches
    ]
    return branch_name in branch_names


def mlflow_monitoring():
    def inner1(func):
        def inner2(*args, **kwargs):
            mlflow_config: MlflowConfig = args[0].mlflow
            mlflow.set_tracking_uri(mlflow_config.tracking_uri)
            repo = git.Repo(REPO_PATH)

            if is_branch_exist("experiments"):
                logger.info("Checking out to 'experiments' branch")
                repo.git.checkout("experiments")
            else:
                logger.info("Creating new branch 'experiments'")
                repo.git.checkout("-b", "experiments")
            if repo.is_dirty():
                logger.info("Auto committing before running")
                repo.git.add(all=True)
                repo.git.commit(message="feat: auto commit")

            mlflow.set_experiment(mlflow_config.experiment_name)

            # print("run_name:", run_name)
            with mlflow.start_run(run_name=mlflow_config.run_name):
                # log param
                if len(args):
                    args_dict = vars(args[0])
                    [
                        mlflow.log_param(k, args_dict[k])
                        for k in args_dict
                        if k != "mlflow"
                    ]

                return func(*args, **kwargs)

        return inner2

    return inner1

# python peripherals
from typing import TypeVar, Type, Union
from pathlib import Path
import shutil
from datetime import datetime

# tap
from tap import Tap

# wandb
import wandb

# git
import git

# deep-signature
from deep_signature.core.base import SeedableObject


class AppArgumentParser(Tap):
    seed: int
    results_base_dir_path: Path


T = TypeVar("T", bound=AppArgumentParser)


def save_tap(dir_path: Path, typed_argument_parser: Tap, file_name: str = 'args.json'):
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / file_name
    typed_argument_parser.save(path=str(file_path))


def save_codebase(dir_path: Path):
    repo = git.Repo('.', search_parent_directories=True)
    codebase_source_dir_path = repo.working_tree_dir
    codebase_destination_dir_path = dir_path / 'code'
    shutil.copytree(src=codebase_source_dir_path, dst=codebase_destination_dir_path, symlinks=True, ignore=shutil.ignore_patterns('.git', '.idea', '__pycache__'))


def _init_app(config: Union[wandb.Config, AppArgumentParser]) -> Path:
    SeedableObject.set_seed(seed=config.seed)
    datetime_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_base_dir_path = Path(config.results_base_dir_path) / Path(datetime_string)

    save_codebase(
        dir_path=results_base_dir_path)

    return results_base_dir_path


def init_app_tap(parser: AppArgumentParser) -> Path:
    results_base_dir_path = _init_app(config=parser)

    save_tap(
        dir_path=results_base_dir_path,
        typed_argument_parser=parser)

    return results_base_dir_path


def init_app_wandb(wandb_config: wandb.Config) -> Path:
    results_base_dir_path = _init_app(config=wandb_config)
    return results_base_dir_path

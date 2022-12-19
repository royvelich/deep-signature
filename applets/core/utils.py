# python peripherals
from typing import TypeVar, Type
from pathlib import Path
import shutil
from datetime import datetime

# tap
from tap import Tap

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
    repo = git.Repo('../..', search_parent_directories=True)
    codebase_source_dir_path = repo.working_tree_dir
    codebase_destination_dir_path = dir_path / 'code'
    shutil.copytree(src=codebase_source_dir_path, dst=codebase_destination_dir_path, symlinks=True, ignore=shutil.ignore_patterns('.git', '.idea', '__pycache__'))


def init_app(typed_argument_parser_class: Type[T]) -> T:
    parser = typed_argument_parser_class().parse_args()
    SeedableObject.set_seed(seed=parser.seed)

    datetime_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser.results_base_dir_path = parser.results_base_dir_path / Path(datetime_string)

    save_tap(
        dir_path=parser.results_base_dir_path,
        typed_argument_parser=parser)

    save_codebase(
        dir_path=parser.results_base_dir_path)

    return parser

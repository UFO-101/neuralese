from pathlib import Path


def repo_path_to_abs_path(path: str | Path) -> Path:
    """
    Convert a path relative to the repository root to an absolute path.

    Args:
        path: A path relative to the repository root.

    Returns:
        The absolute path.
    """
    repo_abs_path = Path(__file__).parent.parent.parent.absolute()
    return repo_abs_path / path


def ensure_dir_exists(dir_path: Path) -> Path:
    """
    Ensure that a directory exists.

    Args:
        dir_path: The path to the directory to ensure exists.

    Returns:
        The path to the directory.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

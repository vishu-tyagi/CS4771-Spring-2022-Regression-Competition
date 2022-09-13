import os
import glob
import logging
import logging.config
from pathlib import Path

import zipfile

from dnn.utils import timing

logger = logging.getLogger(__name__)


@timing
def unzip(file_path: Path, extract_to: Path) -> None:
    """
    Unzip a zip file
    Args:
        file_path (Path): Path to the tar file
        extract_to (Path): Path for the extracted directories
    """
    with zipfile.ZipFile(file_path, 'r') as zip:
        zip.extractall(extract_to)
    return


def collect_files(dirs: list[Path], file_types: list[str]) -> list[Path]:
    """
    Return relevant files from a list of directories
    Args:
        dirs (list[Path]): List of directories to fetch files from
        file_types (list[str]): Type of files to fetch, for example,
        `["*.txt"]` is a common value for this parameter.
    Returns:
        list[Path]: List of paths that point to files found
    """
    all_files = []
    for dir in dirs:
        for file_type in file_types:
            files = glob.glob(os.path.join(dir, file_type))
            all_files += files
    return all_files
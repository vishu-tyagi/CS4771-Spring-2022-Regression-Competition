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

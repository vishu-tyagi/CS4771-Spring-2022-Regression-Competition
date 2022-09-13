import logging
from optparse import Option
from pathlib import Path
from time import time
from functools import wraps
from typing import Optional

import pandas as pd

from dnn.utils.constants import CSV_SEP

logger = logging.getLogger(__name__)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()

        time_taken = te - ts
        hours_taken = time_taken // (60 * 60)
        minutes_taken = time_taken // 60
        seconds_taken = time_taken % 60

        if hours_taken:
            logger.info(f"func:{f.__name__} took: {hours_taken:0.0f} hr and \
                {minutes_taken:0.0f} min")
        elif minutes_taken:
            logger.info(f"func:{f.__name__} took: {minutes_taken:0.0f} min and \
                {seconds_taken:0.2f} sec")
        else:
            logger.info(f"func:{f.__name__} took: {seconds_taken:0.2f} sec")
        return result
    return wrap


def save_csv(
    df: pd.DataFrame,
    save_to: Path
) -> None:
    df.reset_index(inplace=True, drop=True)
    df.to_csv(save_to, sep=CSV_SEP, mode="w+", index=False)


def load_csv(from_: Path) -> pd.DataFrame:
    df = pd.read_csv(from_, sep=CSV_SEP)
    return df
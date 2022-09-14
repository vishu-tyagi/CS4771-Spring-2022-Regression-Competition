import logging
from time import time
from functools import wraps

import pandas as pd

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

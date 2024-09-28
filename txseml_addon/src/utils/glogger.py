'''
Author: George Zhao
Date: 2021-03-20 10:55:09
LastEditors: George Zhao
LastEditTime: 2021-10-02 08:49:25
Description:
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import logging
import os
import time
import sys
sys.path.append('utils')
sys.path.append('.')
try:
    from . import workdir
except:
    import workdir

import functools
import traceback


class Glogger:
    def __init__(self, taskname: str, log_Dir: str, LOG_LEVEL: int = logging.INFO, timestamp: bool = True, std_err: bool = True):
        self.taskname = f"{taskname}_{str(int(time.time() * 1000)) if timestamp == True else ''}"
        self.LOG_LEVEL = LOG_LEVEL
        LOG_NAME = f'{self.taskname}.log'
        if log_Dir is not None:
            if os.path.exists(log_Dir) == False:
                os.mkdir(log_Dir)
            LOG_FILE = os.path.join(
                log_Dir, LOG_NAME)
        else:
            LOG_FILE = False
        LOG_FORMAT = '%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s'

        def set_logger():
            logger = logging.getLogger(LOG_NAME)
            logger.setLevel(LOG_LEVEL)
            formatter = logging.Formatter(
                fmt=LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

            if std_err:
                ch = logging.StreamHandler(sys.stderr)
                ch.setLevel(logging.DEBUG)
                ch.setFormatter(formatter)
                logger.addHandler(ch)

            if LOG_FILE:
                fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
                fh.setLevel(LOG_LEVEL)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            return logger

        self.logger = set_logger()


def log_wrapper(logger: Glogger):
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            result = None
            try:
                result = func(*args, **kw)
            except Exception as e:
                logger.logger.error(
                    f'In {func.__name__}: {repr(e)}: Track: {traceback.format_exc()}')
                raise e
            except Warning as w:
                logger.logger.warning(
                    f'In {func.__name__}: {repr(w)}')
                raise w
            finally:
                return result
        return wrapper
    return decorate

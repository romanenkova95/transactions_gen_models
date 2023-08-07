import logging
from logging import handlers
import os
from typing import Optional


def get_logger(
    log_level: int = logging.INFO,
    name: Optional[str] = None
) -> logging.Logger:
    if not name:
        name = __name__
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
    # handler_stream = logging.StreamHandler()
    # handler_file = handlers.RotatingFileHandler(os.path.join(
    #     'logs',
    #     'module_logs',
    #     f'{name}.log'
    # ))

    # handler_stream.setFormatter(formatter)
    # handler_file.setFormatter(formatter)

    # logger.addHandler(handler_stream)
    # logger.addHandler(handler_file)

    return logger

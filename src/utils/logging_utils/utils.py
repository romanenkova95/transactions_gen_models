"""Logging utils module"""
import logging
from typing import Optional


def get_logger(
    log_level: int = logging.INFO, name: Optional[str] = None
) -> logging.Logger:
    """Method for the logger initializing (unique for each module).

    Args:
    ----
        log_level (int, optional): level of the outputs logs. Defaults to logging.INFO.
        name (Optional[str], optional): Name of the logger
            (will be specified in the output before the message in the square brackets).
            Defaults to None.

    Returns:
    -------
        logging.Logger: logger object.
    """
    if not name:
        name = __name__

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    return logger

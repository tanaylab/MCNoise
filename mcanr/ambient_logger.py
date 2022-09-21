import datetime
import logging
import sys
from typing import Callable
from logging import Logger, StreamHandler
from typing import Any, Optional, IO
from logging import getLogger
import warnings


warnings.filterwarnings(
    "ignore", category=UserWarning, message=r".*Variable names are not unique..*"
)

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=r".*Index.__and__ operating as a set operation.*"
)

warnings.filterwarnings(
    "ignore", category=UserWarning, message=r'.*R object inheriting from "POSIXct".*'
)

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=r".*Support for multi-dimensional indexing .*"
)


# Global logger object.
LOG: Optional[Logger] = None


class LoggingFormatter(logging.Formatter):
    """
    A formatter that uses a decimal point for milliseconds.
    """

    def formatTime(self, record: Any, datefmt: Optional[str] = None) -> str:
        """
        Format the time.
        """
        record_datetime = datetime.datetime.fromtimestamp(record.created)
        if datefmt is not None:
            assert False
            return record_datetime.strftime(datefmt)

        seconds = record_datetime.strftime("%Y-%m-%d %H:%M:%S")
        msecs = round(record.msecs)
        return f"{seconds}.{msecs:03d}"


def setup_logger(
    *,
    level: int = logging.INFO,
    to: IO = sys.stderr,
    time: bool = False,
    name: Optional[str] = None,
) -> Logger:
    """
    Setup the global Logger.
    A second call will fail as the logger will already be set up.

    :param level: The level of messages to write, defaults to logging.INFO
    :type level: int, optional

    :param to: specified, the output, defaults to sys.stderr
    :type to: IO, optional

    :param time: If true, include a millisecond-resolution timestamp in each message. defaults to False
    :type time: bool, optional

    :param name: A name for the logger, will be added to messages, defaults to None
    :type name: Optional[str], optional

    :return: A logger object which is configured as the user requested
    :rtype: Logger
    """
    global LOG
    assert LOG is None, "Logger already exists, call it with logger() function"

    log_format = "%(levelname)s - %(message)s"

    if name is not None:
        log_format = name + " - " + log_format

    if time:
        log_format = "%(asctime)s - " + log_format

    handler = StreamHandler(to)
    handler.setFormatter(LoggingFormatter(log_format))

    LOG = getLogger("AmbientNoise")
    if not LOG.handlers:
        LOG.addHandler(handler)
    LOG.setLevel(level)

    return LOG


def logger() -> Logger:
    """
    Access the global logger.
    If setup_logger has not been called yet, this will call it using the default flags.
    """
    global LOG
    if LOG is None:
        LOG = setup_logger()
    return LOG


def _log_function_start(func: Callable) -> None:
    """Pre function logging"""
    log_obj = logger()
    log_obj.debug("Starting: %s", func.__name__)


def _log_function_end(func: Callable) -> None:
    """Post function logging"""
    log_obj = logger()
    log_obj.debug("Done:  %s", func.__name__)


def logged(
    pre: Callable = _log_function_start, post: Callable = _log_function_end
) -> Callable:
    def decorate(func: Callable) -> Callable:
        def call(*args, **kwargs):
            pre(func)
            result = func(*args, **kwargs)
            post(func)
            return result

        return call

    return decorate

import logging
import sys

__all__ = ['configure_logging', 'clear_logging']

LOG_FORMAT: str = '[%(asctime)-15s] %(levelname)-8s %(message)s'


def configure_logging(level: str = 'INFO',
                      propagate: bool = False,
                      output_stream=sys.stdout,
                      fmt: str = LOG_FORMAT
                      ) -> None:
    """
    Configure the Python logging facility for the `mltk` package.

    Args:
        level: The log level.
        propagate: Whether or not to propagate log messages to parent loggers?
        output_stream: The output stream, where to print the logs.
        fmt: The log message format.
    """
    logger = logging.getLogger('mltk')

    logger.setLevel(level)
    logger.propagate = propagate

    # initialize the handler
    logger.handlers.clear()
    handler = logging.StreamHandler(output_stream)
    handler.setFormatter(logging.Formatter(fmt=fmt))
    logger.addHandler(handler)


def clear_logging() -> None:
    """Clear all logging configs for the `mltk` package."""
    logger = logging.getLogger('mltk')
    logger.propagate = True
    logger.setLevel(logging.NOTSET)
    logger.handlers.clear()


configure_logging()  # configure logging by default settings

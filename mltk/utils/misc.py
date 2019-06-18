import time
from typing import *

import numpy as np
from terminaltables import AsciiTable

from ..config import Config

__all__ = [
    'format_duration', 'ETA', 'minibatch_slices_iterator',
    'optional_apply', 'format_key_values',
]


def format_duration(seconds: Union[float, int],
                    short_units: bool = True,
                    keep_zeros: bool = False):
    """
    Format specified time duration as human readable text.

    >>> format_duration(0)
    '0s'
    >>> format_duration(61)
    '1m 1s'
    >>> format_duration(86400 * 2 + 60)
    '2d 1m'
    >>> format_duration(86400 * 2 + 60, keep_zeros=True)
    '2d 0h 1m 0s'
    >>> format_duration(86400 * 2 + 60, short_units=False)
    '2 days 1 minute'
    >>> format_duration(-1)
    '1s ago'

    Args:
        seconds: Number of seconds of the time duration.
        short_units: Whether or not to use short units ("d", "h", "m", "s")
            instead of long units ("day", "hour", "minute", "second")?
        keep_zeros: Whether or not to keep zero components?
            (e.g., to keep "0h 0m" in "1d 0h 0m 3s").

    Returns:
        str: The formatted time duration.
    """
    if short_units:
        units = [(86400, 'd', 'd'), (3600, 'h', 'h'),
                 (60, 'm', 'm'), (1, 's', 's')]
    else:
        units = [(86400, ' day', ' days'), (3600, ' hour', ' hours'),
                 (60, ' minute', ' minutes'), (1, ' second', ' seconds')]

    if seconds < 0:
        seconds = -seconds
        suffix = ' ago'
    else:
        suffix = ''

    pieces = []
    for uvalue, uname, uname_plural in units[:-1]:
        if seconds >= uvalue:
            val = int(seconds // uvalue)
            pieces.append(f'{val:d}{uname_plural if val > 1 else uname}')
            seconds %= uvalue
        elif keep_zeros and pieces:
            pieces.append(f'0{uname}')

    uname, uname_plural = units[-1][1:]
    if seconds > np.finfo(np.float64).eps:
        pieces.append(f'{seconds:.4g}{uname_plural if seconds > 1 else uname}')
    elif not pieces or keep_zeros:
        pieces.append(f'0{uname}')

    return ' '.join(pieces) + suffix


class ETA(object):
    """
    Class to help compute the Estimated Time Ahead (ETA).

    >>> now = time.time()
    >>> eta = ETA()
    >>> eta.take_snapshot(progress=0.0, now=now)  # record the start time
    >>> eta.get_eta(progress=0.01, now=now + 5.)  # i.e., 1% work costs 5s
    495.0
    """

    def __init__(self):
        """Construct a new :class:`ETA`."""
        self._times = []
        self._progresses = []

    def take_snapshot(self, progress: Union[int, float],
                      now: Optional[Union[int, float]] = None):
        """
        Take a snapshot of ``(progress, now)``, for later computing ETA.

        Args:
            progress: The current progress, range in ``[0, 1]``.
            now: The current timestamp in seconds.  If not specified, use
                ``time.time()``.
        """
        if not self._progresses or progress - self._progresses[-1] > .001:
            # we only record the time and corresponding progress if the
            # progress has been advanced by 0.1%
            if now is None:
                now = time.time()
            self._progresses.append(progress)
            self._times.append(now)

    def get_eta(self,
                progress: Union[int, float],
                now: Optional[Union[int, float]] = None,
                take_snapshot: bool = True) -> Optional[float]:
        """
        Get the Estimated Time Ahead (ETA).

        Args:
            progress: The current progress, range in ``[0, 1]``.
            now: The current timestamp in seconds.  If not specified, use
                ``time.time()``.
            take_snapshot: Whether or not to take a snapshot of
                the specified ``(progress, now)``? (default :obj:`True`)

        Returns:
            The ETA in seconds, or :obj:`None` if the ETA cannot be estimated.
        """
        # TODO: Maybe we can have a better estimation algorithm here!
        if now is None:
            now = time.time()

        if self._progresses:
            time_delta = now - self._times[0]
            progress_delta = progress - self._progresses[0]
            progress_left = 1. - progress
            if progress_delta < 1e-7:
                return None
            eta = time_delta / progress_delta * progress_left
        else:
            eta = None

        if take_snapshot:
            self.take_snapshot(progress, now)

        return eta


def minibatch_slices_iterator(length: int,
                              batch_size: int,
                              skip_incomplete: bool = False
                              ) -> Generator[slice, None, None]:
    """
    Iterate through all the mini-batch slices.

    >>> arr = np.arange(10)
    >>> for batch_s in minibatch_slices_iterator(len(arr), batch_size=4):
    ...     print(arr[batch_s])
    [0 1 2 3]
    [4 5 6 7]
    [8 9]
    >>> for batch_s in minibatch_slices_iterator(
    ...         len(arr), batch_size=4, skip_incomplete=True):
    ...     print(arr[batch_s])
    [0 1 2 3]
    [4 5 6 7]

    Args:
        length: Total length of data in an epoch.
        batch_size: Size of each mini-batch.
        skip_incomplete: If :obj:`True`, discard the final batch if it
            contains less than `batch_size` number of items.

    Yields
        Slices of each mini-batch.  The last mini-batch may contain less
            elements than `batch_size`.
    """
    start = 0
    stop1 = (length // batch_size) * batch_size
    while start < stop1:
        yield slice(start, start + batch_size, 1)
        start += batch_size
    if not skip_incomplete and start < length:
        yield slice(start, length, 1)


def optional_apply(f, value):
    """
    If `value` is not None, return `f(value)`, otherwise return None.

    >>> optional_apply(int, None) is None
    True
    >>> optional_apply(int, '123')
    123

    Args:
        f: The function to apply on `value`.
        value: The value, maybe None.
    """
    if value is not None:
        return f(value)


KEY_VALUES_TYPE = Union[Dict, Config, Iterable[Tuple[str, Any]]]


def format_key_values(key_values: KEY_VALUES_TYPE,
                      title: Optional[str] = None,
                      formatter: Callable[[Any], str] = str,
                      delimiter_char: str = '=') -> str:
    """
    Format key value sequence into str.

    The basic usage, to format a :class:`Config`, a dict or a list of tuples:

    >>> print(format_key_values(Config(a=123, b=Config(value=456))))
    a         123
    b.value   456
    >>> print(format_key_values({'a': 123, 'b': {'value': 456}}))
    a   123
    b   {'value': 456}
    >>> print(format_key_values([('a', 123), ('b', {'value': 456})]))
    a   123
    b   {'value': 456}

    To add a title and a delimiter:

    >>> print(format_key_values(Config(a=123, b=Config(value=456)),
    ...                         title='short title'))
    short title
    =============
    a         123
    b.value   456
    >>> print(format_key_values({'a': 123, 'b': {'value': 456}},
    ...                         title='long long long title'))
    long long long title
    ====================
    a   123
    b   {'value': 456}

    Args:
        key_values: The sequence of key values, may be a :class:`Config`,
            a dict, or a list of (key, value) pairs.
            If it is a :class:`Config`, it will be flatten via
            :meth:`Config.to_flatten_dict()`.
        title: If specified, will prepend a title and a horizontal delimiter
            to the front of returned string.
        formatter: The function to format values.
        delimiter_char: The character to use for the delimiter between title
            and config key values.

    Returns:
        The formatted str.
    """
    if len(delimiter_char) != 1:
        raise ValueError(f'`delimiter_char` must be one character: '
                         f'got {delimiter_char!r}')

    if isinstance(key_values, Config):
        key_values = key_values.to_flatten_dict()

    if hasattr(key_values, 'items'):
        data = [(key, formatter(value)) for key, value in key_values.items()]
    else:
        data = [(key, formatter(value)) for key, value in key_values]

    # use the terminaltables.AsciiTable to format our key values
    table = AsciiTable(data)
    table.padding_left = 0
    table.padding_right = 3
    table.inner_column_border = False
    table.inner_footing_row_border = False
    table.inner_heading_row_border = False
    table.inner_row_border = False
    table.outer_border = False
    lines = [line.rstrip() for line in table.table.split('\n')]

    # prepend a title
    if title is not None:
        max_length = max(max(map(len, lines)), len(title))
        delim = delimiter_char * max_length
        lines = [title, delim] + lines

    return '\n'.join(lines)

import time
from typing import Union, Optional, Generator

import numpy as np

__all__ = [
    'format_duration', 'ETA', 'minibatch_slices_iterator'
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
    >>> eta.get_eta(progress=0.0, now=now)  # record the start time
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

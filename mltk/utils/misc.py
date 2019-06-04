from typing import Union

import numpy as np

__all__ = ['format_duration']


def format_duration(seconds: Union[float, int], short_units: bool = True,
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
            if val > 0 or keep_zeros:
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

from datetime import timedelta
from typing import *

import numpy as np
from terminaltables import AsciiTable

from .config import Config, config_to_dict
from .metrics import MetricStats
from .utils import NOT_SET


__all__ = [
    'format_key_values', 'format_duration', 'MetricsFormatter',
]

KeyValuesType = Union[Dict, Config, Iterable[Tuple[str, Any]]]
RealValue = Union[float, np.ndarray]
MetricValueType = Union[RealValue, Mapping[str, RealValue], MetricStats]
DurationType = Union[float, int, timedelta]


def format_key_values(key_values: KeyValuesType,
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
        key_values = config_to_dict(key_values, flatten=True)

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


def format_duration(duration: DurationType,
                    precision: int = 0) -> str:
    """
    Format given time duration as human readable text.

    >>> format_duration(0)
    '0s'
    >>> format_duration(-1)
    '1s ago'
    >>> format_duration(0.01, precision=2)
    '0.01s'
    >>> format_duration(1.00, precision=2)
    '1s'
    >>> format_duration(1.125)
    '1s'
    >>> format_duration(1.1251, precision=2)
    '1.13s'
    >>> format_duration(1.51)
    '2s'
    >>> format_duration(59.99, precision=2)
    '59.99s'
    >>> format_duration(59.99)
    '1:00'
    >>> format_duration(60)
    '1:00'
    >>> format_duration(61)
    '1:01'
    >>> format_duration(3600)
    '1:00:00'
    >>> format_duration(86400)
    '1d 00:00:00'
    >>> format_duration(86400 + 7200 + 180 + 4)
    '1d 02:03:04'
    >>> format_duration(timedelta(days=1, hours=2, minutes=3, seconds=4))
    '1d 02:03:04'

    Args:
        duration: The number of seconds, or a :class:`timedelta` object.
        precision: Precision of the seconds (i.e., number of digits to print).

    Returns:
        The formatted text.
    """
    if isinstance(duration, timedelta):
        duration = duration.total_seconds()
    else:
        duration = duration
    is_ago = duration < 0
    duration = round(abs(duration), precision)

    def format_time(seconds, pop_leading_zero):
        # first of all, extract the hours and minutes part
        residual = []
        for unit in (3600, 60):
            residual.append(int(seconds // unit))
            seconds = seconds - residual[-1] * unit

        # format the hours and minutes
        segments = []
        for r in residual:
            if not segments and pop_leading_zero:
                if r != 0:
                    segments.append(str(r))
            else:
                segments.append(f'{r:02d}')

        # break seconds into int and real number part
        seconds_int = int(seconds)
        seconds_real = seconds - seconds_int

        # format the seconds
        if segments:
            seconds_int = f'{seconds_int:02d}'
        else:
            seconds_int = str(seconds_int)
        seconds_real = f'{seconds_real:.{precision}f}'.strip('0')
        if seconds_real == '.':
            seconds_real = ''
        seconds_suffix = 's' if not segments else ''
        segments.append(f'{seconds_int}{seconds_real}{seconds_suffix}')

        # now compose the final time str
        return ':'.join(segments)

    if duration < 86400:
        # less then one day, just format the time str as "__:__:__.__"
        ret = format_time(duration, pop_leading_zero=True)
    else:
        # equal or more than one day, format as "__d __:__:__.__"
        days = int(duration // 86400)
        duration = duration - days * 86400
        time_str = format_time(duration, pop_leading_zero=False)
        ret = f'{days}d {time_str}'

    if is_ago:
        ret = f'{ret} ago'

    return ret


class MetricsFormatter(object):

    DELIMETERS: Tuple[str, str] = (': ', '; ')

    def _metric_sort_key(self, name):
        parts = name.split('_')
        prefix_order = {'train': 0, 'val': 1, 'valid': 2, 'test': 3,
                        'predict': 4, 'epoch': 5, 'batch': 6}
        suffix_order = {'time': -1, 'timer': -1}
        return (prefix_order.get(parts[0], 0), suffix_order.get(parts[-1], 0),
                name)

    def _format_value(self, name: str, val: Any) -> str:
        if np.shape(val) == ():
            name_suffix = name.lower().rsplit('_', 1)[-1]
            if name_suffix in ('time', 'timer'):
                return format_duration(val, precision=3)
            else:
                return f'{float(val):.6g}'
        else:
            return str(val)

    def sorted_names(self, names: Sequence[str]) -> List[str]:
        return sorted(names, key=self._metric_sort_key)

    def format_metric(self, name: str, val: Any, sep: str) -> str:
        # if `val` is a dict with "mean" and "std"
        if isinstance(val, dict) and 'mean' in val and \
                (len(val) == 1 or (len(val) == 2 and 'std' in val)):
            mean, std = val['mean'], val.get('std')
        # elif `val` is a MetricStats object
        elif isinstance(val, MetricStats):
            mean, std = val.mean, val.std
        # else we treat `val` as a simple value
        else:
            mean, std = val, None

        # format the value part
        if std is None:
            val_str = self._format_value(name, mean)
        else:
            val_str = f'{self._format_value(name, mean)} ' \
                      f'(±{self._format_value(name, std)})'

        # now construct the final str
        return f'{name}{sep}{val_str}'

    def format(self,
               metrics: Mapping[str, MetricValueType],
               known_names: Optional[Sequence[str]] = None,
               delimeters: Tuple[str, str] = NOT_SET) -> str:
        if delimeters is NOT_SET:
            delimeters = self.DELIMETERS

        buf = []
        name_val_sep, metrics_sep = delimeters
        fmt = lambda name: self.format_metric(name, metrics[name], name_val_sep)

        # format the metrics with known names (thus preserving the known orders)
        for name in (known_names or ()):
            if name in metrics:
                buf.append(fmt(name))

        # format the metrics with unknown names (sorted by `sorted_names`)
        known_names = set(known_names or ())
        for name in self.sorted_names(metrics):
            if name not in known_names:
                buf.append(fmt(name))

        return metrics_sep.join(buf)

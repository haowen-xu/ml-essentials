from typing import *

from terminaltables import AsciiTable

from .config import Config, config_to_dict

__all__ = ['format_key_values']


KeyValuesType = Union[Dict, Config, Iterable[Tuple[str, Any]]]


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

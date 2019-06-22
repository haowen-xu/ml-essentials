from datetime import datetime
from typing import Any, Tuple
from uuid import UUID

import numpy as np
from bson import SON, ObjectId
from bson.json_util import JSONOptions, JSONMode, dumps, loads

__all__ = ['json_dumps', 'json_loads']

JSON_OPTIONS = JSONOptions(json_mode=JSONMode.RELAXED)
JSON_OPTIONS.strict_uuid = False  # do not move it to the constructor above!


def _json_convert(o: Any) -> Any:
    if hasattr(o, 'items'):
        return SON((k, _json_convert(v)) for k, v in o.items())
    elif hasattr(o, '__iter__') and not isinstance(o, (str, bytes, np.ndarray)):
        return list(_json_convert(v) for v in o)
    elif isinstance(o, (np.integer, np.int, np.uint,
                        np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(o)
    elif isinstance(o, (np.float, np.float16, np.float32, np.float64)):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    else:
        return o


def json_dumps(o: Any, *, separators: Tuple[str, str] = (',', ':'),
               allow_nan: bool = False, **kwargs) -> str:
    """
    Serialize the specified object `o` into JSON text.

    This function will convert NumPy arrays into lists, and then call
    :func:`bson.json_util.dumps` to do the remaining things.
    Thus it is compatible with MongoDB types.

    >>> json_dumps(np.concatenate([np.arange(5), [np.nan]], axis=0))
    '[0.0,1.0,2.0,3.0,4.0,{"$numberDouble":"NaN"}]'
    >>> json_dumps({'values': [np.float(0.1), np.int(2)]})
    '{"values":[0.1,2]}'
    >>> json_dumps(datetime(2019, 6, 15, 14, 50))
    '{"$date":"2019-06-15T14:50:00Z"}'
    >>> json_dumps(UUID('b8429bf5-a9c5-44ef-a8a3-f954e8aff204'))
    '{"$uuid":"b8429bf5a9c544efa8a3f954e8aff204"}'
    >>> json_dumps(ObjectId('5d04930e9dcf3fec04050251'))
    '{"$oid":"5d04930e9dcf3fec04050251"}'

    Args:
        o: The object to be serialized.
        separators: The JSON separators, passed to :func:`bson.json_util.dumps`.
        allow_nan: Whether or not to allow `NaN`, `Infinity` in the serialized
            JSON?  Passed to :func:`bson.json_util.dumps`.
        \\**kwargs: Additional arguments and named arguments passed
            to :func:`bson.json_util.dumps`.

    Returns:
        The JSON text.
    """
    return dumps(
        _json_convert(o), json_options=JSON_OPTIONS, separators=separators,
        allow_nan=allow_nan, **kwargs
    )


def json_loads(s: str, **kwargs) -> Any:
    """
    Deserialize the specified JSON text `s` into object.

    This function will call :func:`bson.json_util.loads`, thus is compatible
    with MongoDB types.

    >>> json_loads('[0.0,1.0,2.0,3.0,4.0,{"$numberDouble":"NaN"}]')
    [0.0, 1.0, 2.0, 3.0, 4.0, nan]
    >>> json_loads('{"values":[0.1,2]}')
    {'values': [0.1, 2]}
    >>> json_loads('{"$date":"2019-06-15T14:50:00Z"}')  # doctest: +ELLIPSIS
    datetime.datetime(2019, 6, 15, 14, 50, tzinfo=...)
    >>> json_loads('{"$uuid":"b8429bf5a9c544efa8a3f954e8aff204"}')
    UUID('b8429bf5-a9c5-44ef-a8a3-f954e8aff204')
    >>> json_loads('{"$oid":"5d04930e9dcf3fec04050251"}')
    ObjectId('5d04930e9dcf3fec04050251')

    Args:
        s: The JSON text to be deserialized.
        \\**kwargs: Additional arguments and named arguments passed
            to :func:`bson.json_util.loads`.

    Returns:
        The deserialized object.
    """
    return loads(s, json_options=JSON_OPTIONS, **kwargs)

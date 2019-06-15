import re
from cachetools import LRUCache
from datetime import datetime
from typing import *
from uuid import UUID
from urllib.parse import quote as urlquote

import numpy as np
import requests
from bson import ObjectId
from bson.json_util import SON, JSONOptions, JSONMode, dumps, loads

__all__ = ['MLStorageClient']

JSON_OPTIONS = JSONOptions(json_mode=JSONMode.RELAXED)
JSON_OPTIONS.strict_uuid = False  # do not move it to the constructor above!
DocumentType = FilterType = Dict[str, Any]
IdType = Union[ObjectId, str]


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


_PATH_SEP_SPLITTER = re.compile(r'[/\\]')
_INVALID_PATH_CHARS = re.compile(r'[<>:"|?*]')


def normalize_relpath(path: str) -> str:
    """
    Normalize the `path`, enforcing `path` to be relative, translating "\\"
    into "/", reducing contiguous "/", eliminating "." and "..", and checking
    whether the `path` contains invalid characters.

    >>> normalize_relpath(r'/a/.\\b/c/../d')
    'a/b/d'
    >>> normalize_relpath('c:\\\\windows')
    Traceback (most recent call last):
        ...
    ValueError: Path contains invalid character(s): 'c:\\\\windows'
    >>> normalize_relpath('../')
    Traceback (most recent call last):
        ...
    ValueError: Path jump out of root: '../'

    Args:
        path: The relative path to be normalized.

    Returns:
        The normalized relative path.

    Raises:
        ValueError: If any ".." would jump out of root, or the path contains
            invalid characters.
    """
    if _INVALID_PATH_CHARS.search(path):
        raise ValueError(f'Path contains invalid character(s): {path!r}')
    segments = _PATH_SEP_SPLITTER.split(path)
    ret = []
    for segment in segments:
        if segment == '..':
            try:
                ret.pop()
            except IndexError:
                raise ValueError(f'Path jump out of root: {path!r}')
        elif segment not in ('', '.'):
            ret.append(segment)
    return '/'.join(ret)


class MLStorageClient(object):
    """
    Client binding for MLStorage Server API v1.
    """

    def __init__(self, uri: str):
        """
        Construct a new :class:`ClientV1`.

        Args:
            uri: Base URI of the MLStorage server, e.g., "http://example.com".
        """
        uri = uri.rstrip('/')
        self._uri = uri
        self._storage_dir_cache = LRUCache(128)

    def _update_storage_dir_cache(self, doc):
        self._storage_dir_cache[doc['_id']] = doc['storage_dir']

    @property
    def uri(self) -> str:
        """Get the base URI of the MLStorage server."""
        return self._uri

    def do_request(self, method: str, endpoint: str, decode_json: bool = True,
                   **kwargs) -> Union[requests.Response, Any]:
        """
        Send request of HTTP `method` to given `endpoint`.

        Args:
            method: The HTTP request method.
            endpoint: The endpoint of the API, should start with a slash "/".
                For example, "/_query".
            decode_json: Whether or not to decode the response body as JSON?
            \\**kwargs: Arguments to be passed to :func:`requests.request`.

        Returns:
            The response object if ``decode_json = False``, or the decoded
            JSON object.
        """
        uri = f'{self.uri}/v1{endpoint}'
        if 'json' in kwargs:
            json_obj = kwargs.pop('json')
            json_str = json_dumps(json_obj)
            kwargs['data'] = json_str
            kwargs.setdefault('headers', {})
            kwargs['headers']['Content-Type'] = 'application/json'

        resp = requests.request(method, uri, **kwargs)
        resp.raise_for_status()

        if decode_json:
            content_type = resp.headers.get('content-type') or ''
            content_type = content_type.split(';', 1)[0]
            if content_type != 'application/json':
                raise IOError(f'The response from {uri} is not JSON: '
                              f'HTTP code is {resp.status_code}')
            resp = json_loads(resp.content)

        return resp

    def query(self,
              filter: Optional[FilterType] = None,
              sort: Optional[str] = None,
              skip: int = 0,
              limit: Optional[int] = None) -> List[DocumentType]:
        """
        Query experiment documents according to the `filter`.

        Args:
            filter: The filter dict.
            sort: Sort by which field, a string matching the pattern
                ``[+/-]<field>``.  "+" means ASC order, while "-" means
                DESC order.  For example, "start_time", "+start_time" and
                "-stop_time".
            skip: The number of records to skip.
            limit: The maximum number of records to retrieve.

        Returns:
            The documents of the matched experiments.
        """
        uri = f'/_query?skip={skip}'
        if sort is not None:
            uri += f'&sort={urlquote(sort)}'
        if limit is not None:
            uri += f'&limit={limit}'
        ret = self.do_request('POST', uri, json=filter or {})
        for doc in ret:
            self._update_storage_dir_cache(doc)
        return ret

    def get(self, id: IdType) -> DocumentType:
        """
        Get the document of an experiment by its `id`.

        Args:
            id: The id of the experiment.

        Returns:
            The document of the retrieved experiment.
        """
        ret = self.do_request('GET', f'/_get/{id}')
        self._update_storage_dir_cache(ret)
        return ret

    def heartbeat(self, id: IdType) -> None:
        """
        Send heartbeat packet for the experiment `id`.

        Args:
            id: The id of the experiment.
        """
        self.do_request('POST', f'/_heartbeat/{id}', data=b'')

    def create(self, doc_fields: DocumentType) -> DocumentType:
        """
        Create an experiment.

        Args:
            doc_fields: The document fields of the new experiment.

        Returns:
            The document of the created experiment.
        """
        doc_fields = dict(doc_fields)
        ret = self.do_request('POST', '/_create', json=doc_fields)
        self._update_storage_dir_cache(ret)
        return ret

    def update(self, id: IdType, doc_fields: DocumentType) -> DocumentType:
        """
        Update the document of an experiment.

        Args:
            id: ID of the experiment.
            doc_fields: The fields to be updated.

        Returns:
            The document of the updated experiment.
        """
        ret = self.do_request('POST', f'/_update/{id}', json=doc_fields)
        self._update_storage_dir_cache(ret)
        return ret

    def delete(self, id: IdType) -> List[IdType]:
        """
        Delete an experiment.

        Args:
            id: ID of the experiment.

        Returns:
            List of deleted experiment IDs.
        """
        ret = self.do_request('POST', f'/_delete/{id}', data=b'')
        for i in ret:
            self._storage_dir_cache.pop(i, None)
        return ret

    def set_finished(self,
                     id: IdType,
                     status: str,
                     doc_fields: Optional[DocumentType] = None
                     ) -> DocumentType:
        """
        Set the status of an experiment.

        Args:
            id: ID of the experiment.
            status: The new status, one of {"RUNNING", "COMPLETED", "FAILED"}.
            doc_fields: Optional new document fields to be set.

        Returns:
            The document of the updated experiment.
        """
        doc_fields = dict(doc_fields or ())
        doc_fields['status'] = status
        ret = self.do_request('POST', f'/_set_finished/{id}', json=doc_fields)
        self._update_storage_dir_cache(ret)
        return ret

    def get_storage_dir(self, id: IdType) -> str:
        """
        Get the storage directory of an experiment.

        Args:
            id: ID of the experiment.

        Returns:
            The storage directory of the experiment.
        """
        id = str(id)
        storage_dir = self._storage_dir_cache.get(id, None)
        if storage_dir is None:
            doc = self.get(id)
            storage_dir = doc['storage_dir']
        return storage_dir

    def get_file(self, id: IdType, path: str) -> bytes:
        """
        Get the content of a file in the storage directory of an experiment.

        Args:
            id: ID of the experiment.
            path: Relative path of the file.

        Returns:
            The file content.
        """
        id = str(id)
        path = normalize_relpath(path)
        return self.do_request(
            'GET', f'/_getfile/{id}/{path}', decode_json=False).content

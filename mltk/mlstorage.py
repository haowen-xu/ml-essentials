import re
from cachetools import LRUCache
from typing import *
from urllib.parse import quote as urlquote

import requests
from bson import ObjectId

from .utils import json_dumps, json_loads, DocInherit

__all__ = ['MLStorageClient']

DocumentType = FilterType = Dict[str, Any]
IdType = Union[ObjectId, str]

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

import copy
import os
import re
import time
from contextlib import contextmanager
from typing import *

import numpy as np
from heapdict import heapdict

__all__ = [
    'Singleton', 'NOT_SET',
    'format_duration', 'ETA', 'minibatch_slices_iterator',
    'optional_apply',  'validate_enum_arg',
    'maybe_close', 'iter_files',
    'InheritanceDict', 'CachedInheritanceDict',
    'parse_tags', 'deep_copy',
]


class Singleton(object):
    """
    Base class for singleton classes.

    >>> class Parent(Singleton):
    ...     pass

    >>> class Child(Parent):
    ...     pass

    >>> Parent() is Parent()
    True
    >>> Child() is Child()
    True
    >>> Parent() is not Child()
    True
    """

    __instances_dict = {}

    def __new__(cls, *args, **kwargs):
        if cls not in Singleton.__instances_dict:
            Singleton.__instances_dict[cls] = \
                object.__new__(cls, *args, **kwargs)
        return Singleton.__instances_dict[cls]


class NotSet(Singleton):
    """
    Class of the `NOT_SET` constant.

    >>> NOT_SET is not None
    True
    >>> NOT_SET
    NOT_SET
    >>> NOT_SET == NOT_SET
    True
    >>> NotSet() is NOT_SET
    True
    >>> NotSet() == NOT_SET
    True
    """

    def __repr__(self):
        return 'NOT_SET'


NOT_SET = NotSet()


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


TArgValue = TypeVar('TArgValue')


def validate_enum_arg(arg_name: str,
                      arg_value: Optional[TArgValue],
                      choices: Iterable[TArgValue],
                      nullable: bool = False) -> Optional[TArgValue]:
    """
    Validate the value of an enumeration argument.

    Args:
        arg_name: Name of the argument.
        arg_value: Value of the argument.
        choices: Valid choices of the argument value.
        nullable: Whether or not the argument can be None?

    Returns:
        The validated argument value.

    Raises:
        ValueError: If `arg_value` is not valid.
    """
    choices = tuple(choices)

    if not (nullable and arg_value is None) and (arg_value not in choices):
        raise ValueError('Invalid value for argument `{}`: expected to be one '
                         'of {!r}, but got {!r}.'.
                         format(arg_name, choices, arg_value))

    return arg_value


@contextmanager
def maybe_close(obj):
    """
    Enter a context, and if `obj` has ``.close()`` method, close it
    when exiting the context.

    >>> class HasClose(object):
    ...     def close(self):
    ...         print('closed')

    >>> class HasNotClose(object):
    ...     pass

    >>> with maybe_close(HasClose()) as obj:  # doctest: +ELLIPSIS
    ...     print(obj)
    <mltk.utils.misc.HasClose ...>
    closed

    >>> with maybe_close(HasNotClose()) as obj:  # doctest: +ELLIPSIS
    ...     print(obj)
    <mltk.utils.misc.HasNotClose ...>

    Args:
        obj: The object maybe to close.

    Yields:
        The specified `obj`.
    """
    try:
        yield obj
    finally:
        if hasattr(obj, 'close'):
            obj.close()


def iter_files(root_dir: str, sep: str = '/') -> Generator[str, None, None]:
    """
    Iterate through all files in `root_dir`, returning the relative paths
    of each file.  The sub-directories will not be yielded.

    Args:
        root_dir: The root directory, from which to iterate.
        sep: The separator for the relative paths.

    Yields:
        The relative paths of each file.
    """
    def f(parent_path, parent_name):
        for f_name in os.listdir(parent_path):
            f_child_path = parent_path + os.sep + f_name
            f_child_name = parent_name + sep + f_name
            if os.path.isdir(f_child_path):
                for s in f(f_child_path, f_child_name):
                    yield s
            else:
                yield f_child_name

    for name in os.listdir(root_dir):
        child_path = root_dir + os.sep + name
        if os.path.isdir(child_path):
            for x in f(child_path, name):
                yield x
        else:
            yield name


TValue = TypeVar('TValue')


class _InheritanceNode(object):

    def __init__(self, type_: type):
        self.type = type_
        self.children = []

    def add_child(self, child: '_InheritanceNode'):
        self.children.append(child)


class InheritanceDict(Generic[TValue]):
    """
    A dict that gives the registered value of the closest known ancestor
    of a query type (`ancestor` includes the type itself).

    >>> class GrandPa(object): pass
    >>> class Parent(GrandPa): pass
    >>> class Child(Parent): pass
    >>> class Uncle(GrandPa): pass

    >>> d = InheritanceDict()
    >>> d[Child] = 1
    >>> d[GrandPa] = 2
    >>> d[Uncle] = 3
    >>> d[GrandPa]
    2
    >>> d[Parent]
    2
    >>> d[Child]
    1
    >>> d[Uncle]
    3
    >>> d[str]
    Traceback (most recent call last):
        ...
    KeyError: <class 'str'>
    """

    def __init__(self):
        self._nodes = []  # type: List[_InheritanceNode]
        self._values = {}
        self._topo_sorted = None

    def __setitem__(self, type_: type, value: TValue):
        this_node = _InheritanceNode(type_)
        if type_ not in self._values:
            for node in self._nodes:
                if issubclass(type_, node.type):
                    node.add_child(this_node)
                elif issubclass(node.type, type_):
                    this_node.add_child(node)
            self._nodes.append(this_node)
            self._topo_sorted = None
        self._values[type_] = value

    def __getitem__(self, type_: type) -> TValue:
        if self._topo_sorted is None:
            self._topo_sort()
        for t in reversed(self._topo_sorted):
            if t is type_ or issubclass(type_, t):
                return self._values[t]
        raise KeyError(type_)

    def _topo_sort(self):
        parent_count = {node: 0 for node in self._nodes}
        for node in self._nodes:
            for child in node.children:
                parent_count[child] += 1

        heap = heapdict()
        for node, pa_count in parent_count.items():
            heap[node] = pa_count

        topo_sorted = []
        while heap:
            node, priority = heap.popitem()
            topo_sorted.append(node.type)
            for child in node.children:
                heap[child] -= 1

        self._topo_sorted = topo_sorted


class CachedInheritanceDict(InheritanceDict[TValue]):
    """
    A subclass of :class:`InheritanceDict`, with an additional lookup cache.

    The cache is infinitely large, thus this class is only suitable under the
    situation where the number of queried types are not too large.
    """

    NOT_EXIST = ...

    def __init__(self):
        super().__init__()
        self._cache = {}  # type: Dict[type, TValue]

    def _topo_sort(self):
        self._cache.clear()
        super()._topo_sort()

    def __getitem__(self, type_: type) -> TValue:
        ret = self._cache.get(type_, None)
        if ret is None:
            try:
                ret = self._cache[type_] = super().__getitem__(type_)
            except KeyError:
                self._cache[type_] = self.NOT_EXIST
                raise
        elif ret is self.NOT_EXIST:
            raise KeyError(type_)
        return ret

    def __setitem__(self, type_: type, value: TValue):
        self._cache.clear()
        super().__setitem__(type_, value)


def parse_tags(s: str) -> List[str]:
    """
    Parse comma separated tags str into list of tags.

    >>> parse_tags('one tag')
    ['one tag']
    >>> parse_tags('  strip left and right ends  ')
    ['strip left and right ends']
    >>> parse_tags('two, tags')
    ['two', 'tags']
    >>> parse_tags('"quoted, string" is one tag')
    ['quoted, string is one tag']
    >>> parse_tags(', empty tags,  , will be skipped, ')
    ['empty tags', 'will be skipped']

    Args:
        s: The comma separated tags str.

    Returns:
        The parsed tags.
    """
    tags = []
    buf = []
    in_quoted = None

    for c in s:
        if in_quoted:
            if c == in_quoted:
                in_quoted = None
            else:
                buf.append(c)
        elif c == '"' or c == '\'':
            in_quoted = c
        elif c == ',':
            if buf:
                tag = ''.join(buf).strip()
                if tag:
                    tags.append(tag)
                buf.clear()
        else:
            buf.append(c)

    if buf:
        tag = ''.join(buf).strip()
        if tag:
            tags.append(tag)

    return tags


TValue = TypeVar('TValue')
PatternType = type(re.compile('x'))


def deep_copy(value: TValue) -> TValue:
    """
    A patched deep copy function, that can handle various types cannot be
    handled by the standard :func:`copy.deepcopy`.

    Args:
        value: The value to be copied.

    Returns:
        The copied value.
    """
    def pattern_dispatcher(v, memo=None):
        return v  # we don't need to copy a regex pattern object, it's read-only

    old_dispatcher = copy._deepcopy_dispatch.get(PatternType, None)
    copy._deepcopy_dispatch[PatternType] = pattern_dispatcher
    try:
        return copy.deepcopy(value)
    finally:
        if old_dispatcher is not None:  # pragma: no cover
            copy._deepcopy_dispatch[PatternType] = old_dispatcher
        else:
            del copy._deepcopy_dispatch[PatternType]

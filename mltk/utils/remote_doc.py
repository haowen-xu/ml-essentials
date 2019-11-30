import copy
import time
from abc import ABCMeta, abstractmethod
from enum import Enum
from logging import getLogger
from threading import Thread, Condition, Semaphore
from typing import *

__all__ = [
    'merge_doc_fields', 'RemoteUpdateMode', 'RemoteDoc',
]

DocumentType = Dict[str, Any]


def merge_doc_fields(target: DocumentType,
                     *sources: DocumentType,
                     keys_to_expand: Tuple[str, ...] = (),
                     copy_target: bool = False) -> DocumentType:
    """
    Merge updated fields from `source` dicts into `target`.

    Args:
        target: The target document.
        \\*sources: The source documents.
        keys_to_expand: A list of field names, which are expected to be nested
            dict and should be expanded.
        copy_target: Whether or not to get a copy of the target before merging
            the values from the sources?  Defaults to :obj:`False`.

    Returns:
        The merged updates.
    """
    if copy_target:
        target = copy.copy(target)
    for source in sources:
        if source:
            for key, val in source.items():
                if key in keys_to_expand and isinstance(val, dict):
                    target.pop(key, None)
                    for val_key, val_val in val.items():
                        target[f'{key}.{val_key}'] = val_val
                else:
                    # TODO: do we need to raise error if `key in keys_to_expand`
                    #       but val is not a dict?
                    target[key] = val
    return target


class RemoteUpdateMode(int, Enum):
    """
    Update mode.  Larger number indicates more frequent attempts to push
    updates to the remote.
    """

    STOPPED = 9999  # background thread exited
    NONE = 0  # no pending update
    RETRY = 1  # retry previous failed update
    RELAXED = 2  # update later
    IMMEDIATELY = 3  # update immediately


class RemoteDoc(metaclass=ABCMeta):
    """
    Class that pushes update of a document to remote via a background thread.
    """

    def __init__(self,
                 retry_interval: float,
                 relaxed_interval: float,
                 heartbeat_interval: Optional[float] = None,
                 keys_to_expand: Tuple[str, ...] = ()):
        # parameters
        self.retry_interval: float = retry_interval
        self.relaxed_interval: float = relaxed_interval
        self.heartbeat_interval: Optional[float] = heartbeat_interval
        self.keys_to_expand: Tuple[str, ...] = keys_to_expand

        # the pending updates
        self._updates: Optional[DocumentType] = {}
        self._update_mode: RemoteUpdateMode = RemoteUpdateMode.NONE
        self._last_push_time: float = 0.

        # state of the background worker
        self._thread: Optional[Thread] = None
        self._cond = Condition()
        self._start_sem = Semaphore(0)

    @abstractmethod
    def push_to_remote(self, updates: DocumentType):
        """
        Push pending updates to the remote.

        Args:
            updates: The updates to be pushed to the remote.
        """

    def update(self, fields: DocumentType, immediately: bool = False):
        """
        Queue updates of the remote document into the background worker.

        Args:
            fields: The document updates.
            immediately: Whether or not to let the background worker
                push updates to the remote immediately?  Defaults to
                :obj:`False`, i.e., the updates will be pushed to remote
                later at a proper time.
        """
        with self._cond:
            # set pending updates
            self._updates = merge_doc_fields(
                self._updates, fields,
                keys_to_expand=self.keys_to_expand)

            # set the update mode
            if immediately:
                self._update_mode = RemoteUpdateMode.IMMEDIATELY
            else:
                self._update_mode = RemoteUpdateMode.RELAXED

            # notify the background thread about the new updates
            self._cond.notify_all()

    def flush(self):
        """Push pending updates to the remote in the foreground thread."""
        with self._cond:
            updates = copy.copy(self._updates)
            self._updates.clear()
            if self._update_mode != RemoteUpdateMode.STOPPED:
                self._update_mode = RemoteUpdateMode.NONE
            try:
                self.push_to_remote(updates)
            except:
                self._merge_back(updates, RemoteUpdateMode.RETRY)
                raise

    def _merge_back(self, updates: Optional[DocumentType],
                    mode: RemoteUpdateMode):
        """
        Merge back unsuccessful updates.

        The caller must obtain ``self._cond`` lock before calling this method.

        Args:
            updates: The updates that was not pushed to remote successfully.
            mode: The minimum remote update mode after merged.
        """
        if updates:
            for k, v in updates.items():
                if k not in self._updates:
                    self._updates[k] = v
        if self._update_mode < mode:
            self._update_mode = mode

    def _thread_func(self):
        self._last_push_time = 0.
        self._update_mode = RemoteUpdateMode.NONE

        # notify the main thread that this worker has started
        self._start_sem.release()

        # the main worker loop
        while True:
            # check the update mode and pending updates
            with self._cond:
                mode = self._update_mode
                last_push_time = self._last_push_time
                if mode == RemoteUpdateMode.STOPPED:
                    break

                now_time = time.time()
                elapsed = now_time - last_push_time
                target_itv = {
                    RemoteUpdateMode.IMMEDIATELY: 0,
                    RemoteUpdateMode.RELAXED: self.relaxed_interval,
                    RemoteUpdateMode.RETRY: self.retry_interval,
                    RemoteUpdateMode.NONE: self.heartbeat_interval
                }[mode]

                # if target sleep interval has been reached, we plan to
                # push the updates.  otherwise we plan to do nothing.
                if target_itv is not None:
                    if elapsed >= target_itv:
                        # move the pending updates from the shared status to
                        # the private zone of this thread worker
                        updates = copy.copy(self._updates)
                        self._updates.clear()
                        self._update_mode = RemoteUpdateMode.NONE
                    else:
                        # no plan to execute now, sleep for a bit more
                        self._cond.wait(target_itv - elapsed)
                        continue  # go into next loop to check status again
                else:
                    # no plan to execute, and no sleep interval can be inferred,
                    # wait for an indefinite amount of time
                    self._cond.wait()
                    continue  # go into next loop to check status again

            # now, since the plan has been set, we are about ot execute the plan
            merge_back = False
            if updates is not None:
                try:
                    self.push_to_remote(updates)
                except Exception:
                    getLogger(__name__).warning(
                        'Failed to push updates to remote.', exc_info=True)
                    merge_back = True
                finally:
                    last_push_time = time.time()

            # write back to the shared status
            with self._cond:
                if merge_back:
                    self._merge_back(updates, RemoteUpdateMode.RETRY)
                else:
                    self._merge_back(None, RemoteUpdateMode.NONE)
                self._last_push_time = last_push_time

        # finally, de-reference the thread object
        self._thread = None

    def start_worker(self):
        """Start the background worker."""
        if self._thread is not None:  # pragma: no cover
            raise RuntimeError('Background worker has already started.')

        self._thread = Thread(target=self._thread_func, daemon=True)
        self._thread.start()
        self._start_sem.acquire()

    def stop_worker(self):
        """Stop the background worker."""
        if self._thread is not None:
            thread = self._thread
            with self._cond:
                self._update_mode = RemoteUpdateMode.STOPPED
                self._cond.notify_all()
            thread.join()

    def __enter__(self):
        self.start_worker()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_worker()

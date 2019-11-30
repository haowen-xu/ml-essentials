import time
import unittest

import pytest

from mltk.utils import *
from mltk.utils.remote_doc import DocumentType


class MyError(Exception):
    pass


class MyRemoteDoc(RemoteDoc):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logs = []
        self.raise_error = False

    def push_to_remote(self, updates: DocumentType):
        self.logs.append(deep_copy(updates))
        if self.raise_error:
            time.sleep(0.2)
            raise MyError('an error has occurred')


class RemoteDocTestCase(unittest.TestCase):

    def test_merge_doc_fields(self):
        target = {
            'a': 99, 'c': {'w': 123}, 'd': 100
        }
        sources = [
            {'a': {'x': 1, 'y': 2}, 'b': {'m': 3}},
            {'a': {'x': 4, 'z': 5}, 'c': 6},
            None,
        ]
        expected = {'a': {'x': 4, 'z': 5}, 'b': {'m': 3}, 'c': 6, 'd': 100}
        expected_ex = {'a.x': 4, 'a.y': 2, 'a.z': 5, 'b': {'m': 3}, 'c': 6,
                       'd': 100}

        # test no copy
        target2 = deep_copy(target)
        ret = merge_doc_fields(target2, *sources, keys_to_expand=())
        self.assertIs(ret, target2)
        self.assertEqual(ret, expected)

        target2 = deep_copy(target)
        ret = merge_doc_fields(target2, *sources, keys_to_expand=('a',))
        self.assertIs(ret, target2)
        self.assertEqual(ret, expected_ex)

        # test copy
        ret = merge_doc_fields(target, *sources, keys_to_expand=(),
                               copy_target=True)
        self.assertIsNot(ret, target)
        self.assertEqual(ret, expected)

        ret = merge_doc_fields(target, *sources, keys_to_expand=('a',),
                               copy_target=True)
        self.assertIsNot(ret, target)
        self.assertEqual(ret, expected_ex)

    def test_update(self):
        with MyRemoteDoc(retry_interval=0.5,
                         relaxed_interval=0.2,
                         keys_to_expand=('a',)) as doc:
            # the first request should be finished in time
            doc.update({'a': {'x': 1}, 'b': {'z': 99}})
            pending_updates = {'a.x': 1, 'b': {'z': 99}}
            self.assertEqual(doc._update_mode, RemoteUpdateMode.RELAXED)
            self.assertEqual(doc._updates, pending_updates)
            time.sleep(0.05)
            self.assertEqual(doc.logs, [pending_updates])
            self.assertEqual(doc._update_mode, RemoteUpdateMode.NONE)
            self.assertEqual(doc._updates, {})
            doc.logs.clear()

            # the second request, however, should wait for relaxed_interval
            doc.update({'a': {'x': 2, 'y': 3}})
            pending_updates = {'a.x': 2, 'a.y': 3}
            self.assertEqual(doc._update_mode, RemoteUpdateMode.RELAXED)
            self.assertEqual(doc._updates, pending_updates)
            time.sleep(0.05)
            self.assertEqual(doc.logs, [])
            self.assertEqual(doc._update_mode, RemoteUpdateMode.RELAXED)
            self.assertEqual(doc._updates, pending_updates)
            time.sleep(0.2)
            self.assertEqual(doc.logs, [pending_updates])
            self.assertEqual(doc._update_mode, RemoteUpdateMode.NONE)
            self.assertEqual(doc._updates, {})
            doc.logs.clear()

            # the third request, which is marked as `immediately`, should
            # be executed at once
            doc.update({'b': 123}, immediately=True)
            pending_updates = {'b': 123}
            self.assertEqual(doc._update_mode, RemoteUpdateMode.IMMEDIATELY)
            self.assertEqual(doc._updates, pending_updates)
            time.sleep(0.05)
            self.assertEqual(doc.logs, [pending_updates])
            self.assertEqual(doc._update_mode, RemoteUpdateMode.NONE)
            self.assertEqual(doc._updates, {})
            doc.logs.clear()

            # the forth request, which is not executed before the worker has
            # stopped.
            doc.update({'b': 456})
            pending_updates = {'b': 456}
            self.assertEqual(doc._update_mode, RemoteUpdateMode.RELAXED)
            self.assertEqual(doc._updates, pending_updates)
            self.assertEqual(doc.logs, [])

        # now we've exited, but there is still pending updates to be pushed
        self.assertEqual(doc._update_mode, RemoteUpdateMode.STOPPED)
        self.assertEqual(doc._updates, pending_updates)
        self.assertEqual(doc.logs, [])

        # try to push but failed
        doc.raise_error = True
        with pytest.raises(MyError):
            doc.flush()
        self.assertEqual(doc._update_mode, RemoteUpdateMode.STOPPED)
        self.assertEqual(doc._updates, pending_updates)
        self.assertEqual(doc.logs, [pending_updates])
        doc.logs.clear()

        # try to push but failed, with new mode set (as long as mode != STOPPED)
        doc._update_mode = RemoteUpdateMode.RELAXED
        with pytest.raises(MyError):
            doc.flush()
        self.assertEqual(doc._update_mode, RemoteUpdateMode.RETRY)
        self.assertEqual(doc._updates, pending_updates)
        self.assertEqual(doc.logs, [pending_updates])
        doc.logs.clear()

        # test successful final push
        doc._update_mode = RemoteUpdateMode.STOPPED
        doc.raise_error = False
        doc.flush()
        self.assertEqual(doc._update_mode, RemoteUpdateMode.STOPPED)
        self.assertEqual(doc._updates, {})
        self.assertEqual(doc.logs, [pending_updates])
        doc.logs.clear()

    def test_update_error(self):
        with MyRemoteDoc(retry_interval=0.5,
                         relaxed_interval=0.2,
                         keys_to_expand=('a',)) as doc:
            doc.raise_error = True
            doc.update({'b': 456, 'a': {'x': 2}}, immediately=True)
            self.assertEqual(doc._update_mode, RemoteUpdateMode.IMMEDIATELY)
            self.assertEqual(doc._updates, {'b': 456, 'a.x': 2})

            time.sleep(0.05)
            self.assertEqual(doc.logs, [{'b': 456, 'a.x': 2}])  # push should have taken place
            self.assertEqual(doc._update_mode, RemoteUpdateMode.NONE)
            self.assertEqual(doc._updates, {})
            doc.logs.clear()

            time.sleep(0.16)
            self.assertEqual(doc._update_mode, RemoteUpdateMode.RETRY)  # now the worker has turned into retry mode
            self.assertEqual(doc._updates, {'b': 456, 'a.x': 2})
            doc.logs.clear()

            time.sleep(0.4)
            self.assertEqual(doc.logs, [])  # RETRY mode, not RELAXED mode, still no push happen
            self.assertEqual(doc._update_mode, RemoteUpdateMode.RETRY)  # error has make the worker turn into retry mode
            self.assertEqual(doc._updates, {'a.x': 2, 'b': 456})

            time.sleep(0.11)
            self.assertEqual(doc.logs, [{'a.x': 2, 'b': 456}])  # RETRY request should have been sent this moment
            self.assertEqual(doc._update_mode, RemoteUpdateMode.NONE)
            self.assertEqual(doc._updates, {})
            doc.logs.clear()

            doc.update({'a': {'y': 3}})  # send new update when remote push has not finished
            self.assertEqual(doc._update_mode, RemoteUpdateMode.RELAXED)
            self.assertEqual(doc._updates, {'a.y': 3})
            doc.logs.clear()

            time.sleep(0.21)
            self.assertEqual(doc.logs, [])  # the push request has just finished, and the worker is waiting for next push
            self.assertEqual(doc._update_mode, RemoteUpdateMode.RELAXED)  # RELAXED mode is superior to RETRY
            self.assertEqual(doc._updates, {'a.x': 2, 'a.y': 3, 'b': 456})
            doc.raise_error = False

            time.sleep(0.21)
            self.assertEqual(doc.logs, [{'a.x': 2, 'a.y': 3, 'b': 456}])  # RELAXED mode should have triggered push at this moment
            self.assertEqual(doc._update_mode, RemoteUpdateMode.NONE)
            self.assertEqual(doc._updates, {})

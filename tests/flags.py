import os
import unittest

__all__ = ['slow_test']

FAST = os.environ.get('FAST_TEST', '0') == '1'


def slow_test(method):
    return unittest.skipIf(
        FAST, 'slow tests are skipped in fast test mode')(method)

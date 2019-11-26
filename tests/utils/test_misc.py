import os
import re
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from mltk.utils import *


class HumanizeDurationTestCase(unittest.TestCase):
    ignore_zeros_cases = [
        (0.0, '0 second'),
        (1e-8, '1e-08 second'),
        (0.1, '0.1 second'),
        (1.0, '1 second'),
        (1, '1 second'),
        (1.1, '1.1 seconds'),
        (59, '59 seconds'),
        (59.9, '59.9 seconds'),
        (60, '1 minute'),
        (61, '1 minute 1 second'),
        (62, '1 minute 2 seconds'),
        (119, '1 minute 59 seconds'),
        (120, '2 minutes'),
        (121, '2 minutes 1 second'),
        (122, '2 minutes 2 seconds'),
        (3599, '59 minutes 59 seconds'),
        (3600, '1 hour'),
        (3601, '1 hour 1 second'),
        (3661, '1 hour 1 minute 1 second'),
        (86399, '23 hours 59 minutes 59 seconds'),
        (86400, '1 day'),
        (86401, '1 day 1 second'),
        (172799, '1 day 23 hours 59 minutes 59 seconds'),
        (259199, '2 days 23 hours 59 minutes 59 seconds'),
    ]

    keep_zeros_cases = [
        (0.0, '0 second'),
        (1e-8, '1e-08 second'),
        (0.1, '0.1 second'),
        (1.0, '1 second'),
        (1, '1 second'),
        (1.1, '1.1 seconds'),
        (59, '59 seconds'),
        (59.9, '59.9 seconds'),
        (60, '1 minute 0 second'),
        (61, '1 minute 1 second'),
        (62, '1 minute 2 seconds'),
        (119, '1 minute 59 seconds'),
        (120, '2 minutes 0 second'),
        (121, '2 minutes 1 second'),
        (122, '2 minutes 2 seconds'),
        (3599, '59 minutes 59 seconds'),
        (3600, '1 hour 0 minute 0 second'),
        (3601, '1 hour 0 minute 1 second'),
        (3661, '1 hour 1 minute 1 second'),
        (86399, '23 hours 59 minutes 59 seconds'),
        (86400, '1 day 0 hour 0 minute 0 second'),
        (86401, '1 day 0 hour 0 minute 1 second'),
        (172799, '1 day 23 hours 59 minutes 59 seconds'),
        (259199, '2 days 23 hours 59 minutes 59 seconds'),
    ]

    @staticmethod
    def long_to_short(s):
        for u in ('day', 'hour', 'minute', 'second'):
            s = s.replace(' ' + u + 's', u[0]).replace(' ' + u, u[0])
        return s

    def do_test(self, cases, ago, keep_zeros):
        for seconds, answer in cases:
            if ago:
                seconds = -seconds
                answer = answer + ' ago'
            result = format_duration(seconds, short_units=False,
                                     keep_zeros=keep_zeros)
            self.assertEqual(
                result, answer,
                msg=f'format_duration({seconds!r}, short_units=False, '
                    f'keep_zeros={keep_zeros}) is expected to be {answer!r}, '
                    f'but got {result!r}.'
            )
            result = format_duration(seconds, short_units=True,
                                     keep_zeros=keep_zeros)
            answer = self.long_to_short(answer)
            self.assertEqual(
                result, answer,
                msg=f'format_duration({seconds!r}, short_units=True, '
                    f'keep_zeros={keep_zeros}) is expected to be {answer!r}, '
                    f'but got {result!r}.'
            )

    def test_format_duration(self):
        self.do_test(self.ignore_zeros_cases, ago=False, keep_zeros=False)
        self.do_test(self.ignore_zeros_cases[1:], ago=True, keep_zeros=False)
        self.do_test(self.keep_zeros_cases, ago=False, keep_zeros=True)
        self.do_test(self.keep_zeros_cases[1:], ago=True, keep_zeros=True)


class ETATestCase(unittest.TestCase):

    def test_snapshot(self):
        eta = ETA()
        self.assertListEqual([], eta._times)
        self.assertListEqual([], eta._progresses)

        eta.take_snapshot(0.)
        self.assertEqual(1, len(eta._times))
        self.assertListEqual([0.], eta._progresses)

        eta.take_snapshot(.5)
        self.assertEqual(2, len(eta._times))
        self.assertGreaterEqual(eta._times[1], eta._times[0])
        self.assertListEqual([0., .5], eta._progresses)

        eta.take_snapshot(.50001)
        self.assertEqual(2, len(eta._times))
        self.assertListEqual([0., .5], eta._progresses)

        eta.take_snapshot(1., 12345)
        self.assertEqual(3, len(eta._times))
        self.assertEqual(12345, eta._times[-1])
        self.assertListEqual([0., .5, 1.], eta._progresses)

    def test_get_eta(self):
        self.assertIsNone(ETA().get_eta(0.))

        eta = ETA()
        eta.take_snapshot(0., 0)

        self.assertListEqual([0], eta._times)
        self.assertListEqual([0.], eta._progresses)
        np.testing.assert_allclose(3., eta.get_eta(.25, 1, take_snapshot=False))
        self.assertListEqual([0], eta._times)
        self.assertListEqual([0.], eta._progresses)

        np.testing.assert_allclose(99., eta.get_eta(.01, 1))
        self.assertListEqual([0, 1], eta._times)
        self.assertListEqual([0., .01], eta._progresses)

        np.testing.assert_allclose(57.0, eta.get_eta(.05, 3))
        self.assertListEqual([0, 1, 3], eta._times)
        self.assertListEqual([0., .01, .05], eta._progresses)

    def test_progress_too_small(self):
        eta = ETA()
        eta.take_snapshot(0., 0)
        # progress is too small for estimating the ETA
        self.assertIsNone(eta.get_eta(5e-8, 1.))


class IterFilesTestCase(unittest.TestCase):

    def test_iter_files(self):
        names = ['a/1.txt', 'a/2.txt', 'a/b/1.txt', 'a/b/2.txt',
                 'b/1.txt', 'b/2.txt', 'c.txt']

        with TemporaryDirectory() as tempdir:
            for name in names:
                f_path = os.path.join(tempdir, name)
                f_dir = os.path.split(f_path)[0]
                os.makedirs(f_dir, exist_ok=True)
                with open(f_path, 'wb') as f:
                    f.write(b'')

            self.assertListEqual(names, sorted(iter_files(tempdir)))
            self.assertListEqual(names, sorted(iter_files(tempdir + '/a/../')))


class InheritanceDictTestCase(unittest.TestCase):

    def test_base(self):
        class GrandPa(object): pass
        class Parent(GrandPa): pass
        class Child(Parent): pass
        class Uncle(GrandPa): pass
        class NotExist(object): pass

        d = InheritanceDict()
        d[Child] = 1
        d[GrandPa] = 2
        d[Uncle] = 3

        self.assertEqual(d[GrandPa], 2)
        self.assertEqual(d[Parent], 2)
        self.assertEqual(d[Child], 1)
        self.assertEqual(d[Uncle], 3)

        with pytest.raises(KeyError):
            _ = d[NotExist]

        d[GrandPa] = 22
        self.assertEqual(d[GrandPa], 22)
        self.assertEqual(d[Parent], 22)

        with pytest.raises(KeyError):
            _ = d[NotExist]

    def test_cached(self):
        class GrandPa(object): pass
        class Parent(GrandPa): pass
        class Child(Parent): pass
        class Uncle(GrandPa): pass
        class NotExist(object): pass

        d = CachedInheritanceDict()
        d[Child] = 1
        d[GrandPa] = 2
        d[Uncle] = 3

        self.assertEqual(d[GrandPa], 2)
        self.assertEqual(d[GrandPa], 2)
        self.assertEqual(d[Parent], 2)
        self.assertEqual(d[Parent], 2)
        self.assertEqual(d[Child], 1)
        self.assertEqual(d[Child], 1)
        self.assertEqual(d[Uncle], 3)
        self.assertEqual(d[Uncle], 3)

        with pytest.raises(KeyError):
            _ = d[NotExist]
        with pytest.raises(KeyError):
            _ = d[NotExist]

        d[GrandPa] = 22
        self.assertEqual(d[GrandPa], 22)
        self.assertEqual(d[GrandPa], 22)
        self.assertEqual(d[Parent], 22)
        self.assertEqual(d[Parent], 22)

        with pytest.raises(KeyError):
            _ = d[NotExist]
        with pytest.raises(KeyError):
            _ = d[NotExist]


class DeepCopyTestCase(unittest.TestCase):

    def test_deep_copy(self):
        # test regex
        pattern = re.compile(r'xyz')
        self.assertIs(deep_copy(pattern), pattern)

        # test list of regex
        v = [pattern, pattern]
        o = deep_copy(v)
        self.assertIsNot(v, o)
        self.assertEqual(v, o)
        self.assertIs(v[0], o[0])
        self.assertIs(o[1], o[0])

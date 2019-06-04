import unittest

from mltk.utils import format_duration


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

import unittest

import pytest
from mltk import format_key_values


class FormatKeyValuesTestCase(unittest.TestCase):

    def test_format_key_values(self):
        with pytest.raises(ValueError,
                           match='`delimiter_char` must be one character: '
                                 'got \'xx\''):
            format_key_values({'a': 1}, delimiter_char='xx')

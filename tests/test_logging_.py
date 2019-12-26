import logging
import sys
import unittest
from logging import getLogger, StreamHandler

from mltk import *


class LoggingTestCase(unittest.TestCase):

    def test_configure(self):
        logger = getLogger('mltk')

        configure_logging('DEBUG', True, sys.stderr)
        self.assertIsInstance(logger.handlers[0], StreamHandler)
        self.assertIs(logger.handlers[0].stream, sys.stderr)
        self.assertEqual(logger.level, logging.DEBUG)
        self.assertTrue(logger.propagate)

        clear_logging()
        self.assertEqual(len(logger.handlers), 0)
        self.assertEqual(logger.level, logging.NOTSET)
        self.assertTrue(logger.propagate)

        configure_logging()
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], StreamHandler)
        self.assertIs(logger.handlers[0].stream, sys.stdout)
        self.assertEqual(logger.level, logging.INFO)
        self.assertFalse(logger.propagate)

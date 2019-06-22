import os
import sys
import unittest
from tempfile import TemporaryDirectory

import pytest

from mltk import *
from tests.helpers import set_environ_context


class _YourConfig(Config):
    max_epoch = 100
    learning_rate = 0.123


class ExperimentTestCase(unittest.TestCase):

    def test_construct(self):
        # test auto select script name and output dir
        exp = Experiment(_YourConfig)
        script_name = os.path.splitext(
            os.path.basename(sys.modules['__main__'].__file__))[0]
        self.assertEqual(exp.script_name, script_name)
        self.assertEqual(
            exp.output_dir, os.path.abspath(f'./results/{script_name}'))

        # test select output dir according to mlrunner env
        with TemporaryDirectory() as temp_dir:
            with set_environ_context(MLSTORAGE_OUTPUT_DIR=temp_dir):
                exp = Experiment(_YourConfig)
                self.assertEqual(exp.output_dir, temp_dir)

        # test config
        self.assertIsInstance(Experiment(_YourConfig).config, _YourConfig)
        config = _YourConfig()
        self.assertIs(Experiment(config).config, config)

        with pytest.raises(TypeError,
                           match='`config_or_cls` is neither a Config class, '
                                 'nor a Config instance: <class \'object\'>'):
            _ = Experiment(object)

        with pytest.raises(TypeError,
                           match='`config_or_cls` is neither a Config class, '
                                 'nor a Config instance: <object .*>'):
            _ = Experiment(object())

        # test result
        self.assertDictEqual(Experiment(_YourConfig).results, {})

        # test args
        self.assertIsNone(Experiment(_YourConfig).args)
        args = ('--output-dir=abc', '--max_epoch=123')
        self.assertTupleEqual(Experiment(_YourConfig, args=args).args, args)

    def test_context(self):
        with TemporaryDirectory() as temp_dir:
            pass

# -*- coding: utf-8 -*-

import os
import shutil
import sys
import unittest
from tempfile import TemporaryDirectory

import pytest

from mltk import *
from mltk.utils import json_loads
from tests.helpers import set_environ_context, get_file_content, zip_snapshot


@config_params(undefined_fields=True)
class _YourConfig(Config):
    max_epoch = 100
    learning_rate = 0.5

    class train(Config):
        batch_size = 64


class ExperimentTestCase(unittest.TestCase):

    def test_construct(self):
        # test auto select script name and output dir
        exp = Experiment(_YourConfig)
        script_name = os.path.splitext(
            os.path.basename(sys.modules['__main__'].__file__))[0]
        self.assertEqual(exp.script_name, script_name)
        self.assertEqual(
            exp.output_dir, os.path.abspath(f'./results/{script_name}'))
        self.assertIsNone(exp.id)
        self.assertIsNone(exp.client)

        # test select output dir according to mlrunner env
        with TemporaryDirectory() as temp_dir:
            with set_environ_context(MLSTORAGE_OUTPUT_DIR=temp_dir):
                exp = Experiment(_YourConfig)
                self.assertEqual(exp.output_dir, temp_dir)

        # test specifying script_name
        exp = Experiment(_YourConfig, script_name='abc')
        self.assertEqual(exp.script_name, 'abc')
        self.assertEqual(exp.output_dir, os.path.abspath(f'./results/abc'))

        # test specifying the output dir
        with TemporaryDirectory() as temp_dir:
            exp = Experiment(_YourConfig, output_dir=temp_dir)
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
        self.assertTupleEqual(Experiment(_YourConfig).args, tuple(sys.argv[1:]))
        self.assertIsNone(Experiment(_YourConfig, args=None).args)

        args = ('--output-dir=abc', '--max_epoch=123')
        self.assertTupleEqual(Experiment(_YourConfig, args=args).args, args)

        # test specifying the environment variable of MLStorage server
        with set_environ_context(MLSTORAGE_SERVER_URI='http://localhost:8080',
                                 MLSTORAGE_EXPERIMENT_ID='your_id'):
            e = Experiment(_YourConfig)
            self.assertEqual(e.id, 'your_id')
            self.assertIsInstance(e.client, MLStorageClient)
            self.assertEqual(e.client.uri, 'http://localhost:8080')

    def test_events(self):
        with TemporaryDirectory() as temp_dir:
            event_track = []

            exp = Experiment(_YourConfig, output_dir=temp_dir, args=None)
            exp.on_enter.do(lambda: event_track.append('on enter'))
            exp.on_exit.do(lambda: event_track.append('on exit'))

            event_track.append('before enter')
            with exp:
                event_track.append('after enter')
                event_track.append('before exit')
            event_track.append('after exit')

            self.assertListEqual(event_track, [
                'before enter', 'on enter', 'after enter',
                'before exit', 'on exit', 'after exit',
            ])

    def test_experiment(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, 'output')

            # test new output dir
            with Experiment(_YourConfig, output_dir=output_dir,
                            args=('--max_epoch=200', '--train.batch_size=128')
                            ) as exp:
                self.assertEqual(exp.output_dir, output_dir)
                self.assertTrue(os.path.isdir(exp.output_dir))

                # the config should be correctly loaded from CLI argument,
                # and the values should be saved into "/config.json"
                self.assertEqual(exp.config, _YourConfig(
                    max_epoch=200, train=_YourConfig.train(batch_size=128)
                ))
                self.assertDictEqual(
                    json_loads(get_file_content(
                        os.path.join(exp.output_dir, 'config.json'))),
                    {'max_epoch': 200, 'learning_rate': 0.5,
                     'train.batch_size': 128}
                )
                self.assertDictEqual(
                    json_loads(get_file_content(
                        os.path.join(exp.output_dir, 'config.defaults.json'))),
                    {'max_epoch': 100, 'learning_rate': 0.5,
                     'train.batch_size': 64}
                )

                # there should have not been "/result.json"
                self.assertDictEqual(exp.results, {})
                self.assertFalse(os.path.exists(
                    os.path.join(output_dir, 'result.json')))

                # after we `update_results`, there should be "/result.json"
                exp.update_results({'loss': 1, 'acc': 2})
                self.assertDictEqual(exp.results, {'loss': 1, 'acc': 2})
                self.assertDictEqual(
                    json_loads(get_file_content(
                        os.path.join(exp.output_dir, 'result.json'))),
                    {'loss': 1, 'acc': 2}
                )

                # further modify `config` and `results`, and they should be
                # saved after exiting the context
                exp.config.max_epoch = 300
                exp.config.max_step = 10000
                exp.results['acc'] = 3
                exp.results['time'] = 4

                # test to take absolute path
                self.assertEqual(exp.abspath('a/b/c.txt'),
                                 os.path.join(output_dir, 'a/b/c.txt'))

                # test to make directory
                self.assertFalse(os.path.exists(os.path.join(output_dir, 'a')))
                self.assertEqual(exp.make_dirs('a/b'),
                                 os.path.join(output_dir, 'a/b'))
                self.assertTrue(os.path.isdir(os.path.join(output_dir, 'a/b')))
                exp.make_dirs('a/b')

                with pytest.raises(IOError, match='.* exists'):
                    exp.make_dirs('a/b', exist_ok=False)

                # test to make parents
                self.assertFalse(os.path.exists(os.path.join(output_dir, 'b')))
                self.assertEqual(exp.make_parent('b/c.txt'),
                                 os.path.join(output_dir, 'b/c.txt'))
                self.assertTrue(os.path.isdir(os.path.join(output_dir, 'b')))

                # test to open file
                with pytest.raises(IOError, match='No such file or directory'):
                    _ = exp.open_file('c/d.txt', 'wb', make_parent=False)

                with pytest.raises(IOError, match='No such file or directory'):
                    _ = exp.open_file('c/d.txt', 'rb')

                self.assertFalse(os.path.exists(os.path.join(output_dir, 'c')))

                with exp.open_file('c/d.txt', 'wb') as f:
                    f.write(b'hello, world!')
                self.assertEqual(
                    get_file_content(os.path.join(output_dir, 'c/d.txt')),
                    b'hello, world!'
                )

                with exp.open_file('c/d.txt', 'wb', encoding='gbk') as f:
                    f.write('你好，世界')
                self.assertEqual(
                    get_file_content(os.path.join(output_dir, 'c/d.txt')),
                    b'\xc4\xe3\xba\xc3\xa3\xac\xca\xc0\xbd\xe7'
                )

                with exp.open_file('c/d.txt', 'ab') as f:
                    f.write(b'\nhi, world!')

                with exp.open_file('c/d.txt', 'rb', 'gbk') as f:
                    cnt = f.read()
                self.assertIsInstance(cnt, str)
                self.assertEqual(cnt, '你好，世界\nhi, world!')

                # test to put & get file content
                exp.put_file_content(
                    'c/nested/e.txt', '你好，世界', encoding='gbk')
                exp.put_file_content(
                    'c/nested/e.txt', b'\nhello, world!', append=True)
                self.assertEqual(exp.get_file_content('c/nested/e.txt'),
                                 b'\xc4\xe3\xba\xc3\xa3\xac\xca\xc0\xbd\xe7'
                                 b'\nhello, world!')

                # test make_archive error
                with pytest.raises(IOError, match='Not a directory'):
                    _ = exp.make_archive('non-exist')

                # test make_archive without deleting source directory
                zip_file = os.path.join(output_dir, 'cc.zip')
                self.assertEqual(
                    exp.make_archive('c', zip_file, delete_source=False),
                    zip_file
                )
                self.assertTrue(os.path.isdir(os.path.join(output_dir, 'c')))
                self.assertDictEqual(zip_snapshot(zip_file), {
                    'd.txt': b'\xc4\xe3\xba\xc3\xa3\xac\xca\xc0\xbd\xe7'
                             b'\nhi, world!',
                    'nested': {
                        'e.txt': b'\xc4\xe3\xba\xc3\xa3\xac\xca\xc0\xbd\xe7'
                                 b'\nhello, world!'
                    }
                })

                # test make archive with deleting source directory,
                # and merging the original archive files
                old_zip_file = zip_file
                zip_file = os.path.join(output_dir, 'c.zip')
                shutil.move(old_zip_file, zip_file)

                os.remove(os.path.join(output_dir, 'c/nested/e.txt'))
                exp.put_file_content('c/d.txt', b'd.txt')
                exp.put_file_content('c/nested/f.txt', b'f.txt')

                self.assertEqual(exp.make_archive('c'), zip_file)
                self.assertFalse(os.path.exists(os.path.join(output_dir, 'c')))
                self.assertDictEqual(zip_snapshot(zip_file), {
                    'd.txt': b'd.txt',
                    'nested': {
                        'e.txt': b'\xc4\xe3\xba\xc3\xa3\xac\xca\xc0\xbd\xe7'
                                 b'\nhello, world!',
                        'f.txt': b'f.txt',
                    },
                })

                # test make archive on exit
                exp.put_file_content('c/nested/f.txt', b'overrided f.txt')
                exp.put_file_content('c/nested/g.txt', b'g.txt')
                exp.make_archive_on_exit('c')

            # check whether the config and results have been saved
            self.assertDictEqual(
                json_loads(get_file_content(
                    os.path.join(exp.output_dir, 'config.json'))),
                {'max_epoch': 300, 'max_step': 10000, 'learning_rate': 0.5,
                 'train.batch_size': 128}
            )
            self.assertDictEqual(
                json_loads(get_file_content(
                    os.path.join(exp.output_dir, 'result.json'))),
                {'loss': 1, 'acc': 3, 'time': 4}
            )

            # check whether the archive has been made on exit
            zip_file = os.path.join(output_dir, 'c.zip')
            self.assertFalse(os.path.exists(os.path.join(output_dir, 'c')))
            self.assertDictEqual(zip_snapshot(zip_file), {
                'd.txt': b'd.txt',
                'nested': {
                    'e.txt': b'\xc4\xe3\xba\xc3\xa3\xac\xca\xc0\xbd\xe7'
                             b'\nhello, world!',
                    'f.txt': b'overrided f.txt',
                    'g.txt': b'g.txt',
                },
            })

            # test restore from the previous output dir
            # (and also parse `output_dir` from CLI arguments)
            exp = Experiment(_YourConfig, args=['--output-dir=' + output_dir])
            self.assertNotEqual(exp.output_dir, output_dir)
            with exp:
                self.assertEqual(exp.output_dir, output_dir)
                self.assertEqual(exp.config, _YourConfig(
                    max_epoch=300, max_step=10000, train=_YourConfig.train(
                        batch_size=128
                    )
                ))

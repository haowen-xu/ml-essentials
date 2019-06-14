# -*- encoding: utf-8 -*-

import os
import re
import unittest
from functools import partial
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from mltk import *
from mltk.training import TrainLoopCheckpointManager, TrainLoopState


def assert_match_logs(logs, expected):
    if isinstance(logs, list):
        logs2 = '\n'.join(logs)
    else:
        logs2 = logs

    if isinstance(expected, list):
        expected2 = '\n'.join(expected)
    else:
        expected2 = expected

    expected2 = '.*?'.join(map(re.escape, expected2.split('...')))
    expected2 = f'^{expected2}$'

    if not re.match(expected2, logs2, re.M):
        raise AssertionError(f'`logs` not match `expected`: '
                             f'{logs!r} vs {expected!r}')


class TrainLoopFreqTestCase(unittest.TestCase):

    def test_equality(self):
        freq1 = TrainLoopFreq(epochs=1)
        freq2 = TrainLoopFreq(epochs=1)
        self.assertEqual(freq1, freq2)
        self.assertNotEqual(freq1, TrainLoopFreq(epochs=2))
        self.assertEqual(hash(freq1), hash(freq2))

    def test_parse(self):
        with pytest.raises(TypeError,
                           match='`freq` is neither a str nor a TrainLoopFreq: '
                                 '<object .*>'):
            _ = TrainLoopFreq.parse(object())

        freq = TrainLoopFreq(steps=2)
        self.assertIs(TrainLoopFreq.parse(freq), freq)

    def test_constants(self):
        self.assertEqual(TrainLoopFreq.NEVER, TrainLoopFreq())
        self.assertEqual(TrainLoopFreq.EVERY_EPOCH, TrainLoopFreq(epochs=1))
        self.assertEqual(TrainLoopFreq.EVERY_STEP, TrainLoopFreq(steps=1))

    def test_register_callback(self):
        # test every never
        loop = TrainLoop()
        logs = []
        cb = lambda: logs.append('never')
        TrainLoopFreq.NEVER.register_callback(loop, cb)
        self.assertTrue(len(loop.on_exit_epoch._callbacks) == 0)
        self.assertTrue(len(loop.on_exit_step._callbacks) == 0)

        # test every epoch
        loop = TrainLoop()
        logs = []
        cb = lambda: logs.append('every epoch')
        TrainLoopFreq(epochs=1).register_callback(loop, cb)
        self.assertTrue(len(loop.on_exit_epoch._callbacks) == 1)
        self.assertTrue(len(loop.on_exit_step._callbacks) == 0)
        self.assertIs(loop.on_exit_epoch._callbacks[-1], cb)

        # test every step
        loop = TrainLoop()
        logs = []
        cb = lambda: logs.append('every step')
        TrainLoopFreq(steps=1).register_callback(loop, cb)
        self.assertTrue(len(loop.on_exit_epoch._callbacks) == 0)
        self.assertTrue(len(loop.on_exit_step._callbacks) == 1)
        self.assertIs(loop.on_exit_step._callbacks[-1], cb)

        # the other cases are tested in `TrainLoopTestCase`


class TrainLoopCheckpointManagerTestCase(unittest.TestCase):

    def test_checkpoint(self):
        with TemporaryDirectory() as temp_dir:
            temp_dir = os.path.join(temp_dir, 'checkpoints')

            # construction argument validation
            state = TrainLoopState()
            memo = SimpleStatefulObject()
            a = SimpleStatefulObject()
            b = SimpleStatefulObject()
            objects = {'a': a, 'b': b}

            with pytest.raises(ValueError,
                               match='`max_to_keep` must be at least 1: got 0'):
                _ = TrainLoopCheckpointManager(
                    temp_dir, state, memo, objects, max_to_keep=0)

            # test load non-exist checkpoint dir
            saver = TrainLoopCheckpointManager(
                temp_dir, state, memo, objects, max_to_keep=2)
            with pytest.raises(IOError,
                               match=f'`checkpoint_dir` not exist: {temp_dir}'):
                saver.restore(os.path.join(temp_dir, '0'))

            # test save
            saver = TrainLoopCheckpointManager(
                temp_dir, state, memo, objects, max_to_keep=None)
            self.assertIsNone(saver.latest_checkpoint_dir())
            self.assertFalse(os.path.isdir(os.path.join(temp_dir, '0')))

            a.value = 1230
            b.value = 4560
            memo.note = 'hello0'

            saver.save()
            self.assertTrue(os.path.isdir(os.path.join(temp_dir, '0')))

            # test load
            state = TrainLoopState()
            memo = SimpleStatefulObject()
            a = SimpleStatefulObject()
            b = SimpleStatefulObject()
            objects = {'a': a, 'b': b}

            saver = TrainLoopCheckpointManager(
                temp_dir, state, memo, objects, max_to_keep=2)
            self.assertEqual(saver.latest_checkpoint_dir(),
                             os.path.join(temp_dir, '0'))
            saver.restore(saver.latest_checkpoint_dir())

            self.assertEqual(state.epoch, 0)
            self.assertEqual(state.step, 0)
            self.assertEqual(a.value, 1230)
            self.assertEqual(b.value, 4560)
            self.assertEqual(memo.note, 'hello0')

            # make invalid files and directories to test saver behavior
            os.makedirs(os.path.join(temp_dir, 'invalid'))
            with open(os.path.join(temp_dir, '999'), 'wb') as f:
                f.write(b'')

            # test purge old and save error
            for i in range(1, 3):
                state.epoch = i
                state.step = i * 10
                a.value = 1230 + i
                b.value = 4560 + i
                memo.note = f'hello{i}'

                saver.save()
                saver = TrainLoopCheckpointManager(
                    temp_dir, state, memo, objects, max_to_keep=2)

            os.makedirs(os.path.join(
                temp_dir, f'30/{TrainLoopCheckpointManager.OBJECTS_FILE_NAME}'))

            with pytest.raises(IOError,
                               match='Is a directory: \'.*objects.npz\''):
                state.step = 30
                saver.save()

            self.assertFalse(os.path.isdir(os.path.join(temp_dir, '0')))
            self.assertTrue(os.path.isdir(os.path.join(temp_dir, '10')))
            self.assertTrue(os.path.isdir(os.path.join(temp_dir, '20')))

            # test restore
            state = TrainLoopState()
            memo = SimpleStatefulObject()
            a = SimpleStatefulObject()
            b = SimpleStatefulObject()
            objects = {'a': a, 'b': b}

            saver = TrainLoopCheckpointManager(
                temp_dir, state, memo, objects, max_to_keep=2)

            saver.restore(saver.latest_checkpoint_dir())
            self.assertEqual(state.epoch, 2)
            self.assertEqual(state.step, 20)
            self.assertEqual(a.value, 1232)
            self.assertEqual(b.value, 4562)
            self.assertEqual(memo.note, 'hello2')

            saver.restore(os.path.join(temp_dir, '10'))
            self.assertEqual(state.epoch, 1)
            self.assertEqual(state.step, 10)
            self.assertEqual(a.value, 1231)
            self.assertEqual(b.value, 4561)
            self.assertEqual(memo.note, 'hello1')

            # test saver without objects
            state = TrainLoopState()
            memo = SimpleStatefulObject()
            saver = TrainLoopCheckpointManager(
                temp_dir, state, memo, max_to_keep=2)

            saver.restore(saver.latest_checkpoint_dir())
            self.assertEqual(state.epoch, 2)
            self.assertEqual(state.step, 20)
            self.assertEqual(memo.note, 'hello2')

            state.epoch = 4
            state.step = 40
            memo.note = 'hello4'
            saver.save()

            state = TrainLoopState()
            memo = SimpleStatefulObject()
            saver = TrainLoopCheckpointManager(
                temp_dir, state, memo, max_to_keep=2)
            saver.restore(saver.latest_checkpoint_dir())
            self.assertEqual(state.epoch, 4)
            self.assertEqual(state.step, 40)
            self.assertEqual(memo.note, 'hello4')

            # test restore error
            os.makedirs(os.path.join(
                temp_dir, f'40/{TrainLoopCheckpointManager.OBJECTS_FILE_NAME}'))

            state = TrainLoopState(epoch=1, step=10)
            memo = SimpleStatefulObject()
            memo.note = 'hello1'
            a = SimpleStatefulObject()
            b = SimpleStatefulObject()
            objects = {'a': a, 'b': b}

            saver = TrainLoopCheckpointManager(
                temp_dir, state, memo, objects, max_to_keep=2)

            with pytest.raises(IOError,
                               match='Is a directory: \'.*objects.npz\''):
                saver.restore(os.path.join(temp_dir, '40'))

            self.assertEqual(state.epoch, 1)
            self.assertEqual(state.step, 10)
            self.assertEqual(memo.note, 'hello1')


class TrainLoopTestCase(unittest.TestCase):

    def test_interface(self):
        # empty loop
        loop = TrainLoop()
        self.assertIsNone(loop.objects)
        self.assertEqual(loop.epoch, 0)
        self.assertEqual(loop.step, 0)
        self.assertIsInstance(loop.memo, SimpleStatefulObject)
        self.assertIsNone(loop.max_epoch)
        self.assertIsNone(loop.max_step)
        self.assertEqual(loop.log_freq, TrainLoopFreq.EVERY_EPOCH)
        self.assertEqual(loop.checkpoint_freq, TrainLoopFreq.NEVER)
        self.assertIsNone(loop.checkpoint_root)
        self.assertIsNone(loop.get_progress())
        self.assertIsNone(loop.get_eta())

        with pytest.raises(RuntimeError,
                           match='Checkpoint directory is not configured.'):
            loop.make_checkpoint()

        context_required_msg = 'Neither an epoch nor a step loop has been ' \
                               'entered.'
        with loop:
            with pytest.raises(RuntimeError, match=context_required_msg):
                loop.collect(loss=1.)

            with pytest.raises(RuntimeError, match=context_required_msg):
                loop.print_stats()

            with pytest.raises(RuntimeError, match=context_required_msg):
                with loop.timeit('test_time'):
                    pass

            with pytest.raises(RuntimeError, match=context_required_msg):
                with loop.collector('test_acc'):
                    pass

            with pytest.raises(RuntimeError,
                               match='A disposable context cannot be entered '
                                     'twice'):
                with loop:
                    pass

        # set the max_epoch and max_step
        loop.max_epoch = 10
        self.assertEqual(loop.max_epoch, 10)
        self.assertEqual(loop.get_progress(), 0.)

        loop.max_step = 100
        self.assertEqual(loop.max_step, 100)
        self.assertEqual(loop.get_progress(), 0.)

        # test construct with arguments
        with TemporaryDirectory() as temp_dir:
            logs = []
            objects = SimpleStatefulObject()
            loop = TrainLoop(
                objects=objects,
                max_epoch=10,
                max_step=100,
                log_freq=TrainLoopFreq.NEVER,
                checkpoint_root=temp_dir,
                checkpoint_freq='every 100 epochs',
                print_func=logs.append,
            )

            self.assertIs(loop.objects, objects)
            self.assertEqual(loop.max_epoch, 10)
            self.assertEqual(loop.max_step, 100)
            self.assertEqual(loop.log_freq, TrainLoopFreq.NEVER)
            self.assertEqual(loop.checkpoint_root, temp_dir)
            self.assertEqual(loop.checkpoint_freq, TrainLoopFreq(epochs=100))
            loop.print('hello')
            self.assertListEqual(logs, ['hello'])

    def test_loop(self):
        def data_generator(limit):
            for i in range(limit):
                yield i

        # test basic loop without max_epoch or max_step, but with limit
        logs = []
        collected = []

        with TrainLoop(print_func=logs.append) as loop:
            for epoch in loop.iter_epochs(limit=2):
                for step, batch in loop.iter_steps(data_generator(5),
                                                   limit=3 * epoch):
                    collected.append((epoch, step, batch))
                    loop.collect(batch=batch)
                loop.collect(test=epoch)

        assert_match_logs(logs, [
            '[Epoch 1, Step 3] epoch_time: ...s; step_time: ...s (±...s); '
            'batch: 1 (±0.816497); test: 1',
            '[Epoch 2, Step 6] epoch_time: ...s; step_time: ...s (±...s); '
            'batch: 1 (±0.816497); test: 2'
        ])
        self.assertListEqual(collected, [
            (1, 1, 0), (1, 2, 1), (1, 3, 2), (2, 4, 0), (2, 5, 1), (2, 6, 2)
        ])

        # test basic loop without max_epoch or max_step, but with count
        logs = []
        collected = []

        with TrainLoop(log_freq='every 2 steps',
                       print_func=logs.append) as loop:
            for epoch in loop.iter_epochs(count=2):
                for step, batch in loop.iter_steps(data_generator(5),
                                                   count=3):
                    collected.append((epoch, step, batch))
                    loop.collect(batch=batch)

        assert_match_logs(logs, [
            '[Epoch 1, Step 2] step_time: ...s (±...s); batch: 0.5 (±0.5)',
            '[Epoch 2, Step 4] step_time: ...s (±...s); batch: 1 (±1)',
            '[Epoch 2, Step 6] step_time: ...s (±...s); batch: 1.5 (±0.5)',
        ])
        self.assertListEqual(collected, [
            (1, 1, 0), (1, 2, 1), (1, 3, 2), (2, 4, 0), (2, 5, 1), (2, 6, 2)
        ])

        # test basic loop with max_epoch
        logs = []
        collected = []

        with TrainLoop(max_epoch=2, print_func=logs.append) as loop:
            for epoch in loop.iter_epochs():
                for step, batch in loop.iter_steps(data_generator(3)):
                    collected.append((epoch, step, batch))
                    loop.collect(batch=batch)
                loop.collect(test=epoch)

        assert_match_logs(logs, [
            '[Epoch 1/2, Step 3, ETA ...s] epoch_time: ...s; step_time: '
            '...s (±...s); batch: 1 (±0.816497); test: 1',
            '[Epoch 2/2, Step 6, ETA ...s] epoch_time: ...s; step_time: '
            '...s (±...s); batch: 1 (±0.816497); test: 2'
        ])
        self.assertListEqual(collected, [
            (1, 1, 0), (1, 2, 1), (1, 3, 2), (2, 4, 0), (2, 5, 1), (2, 6, 2)
        ])

        # test basic loop with max_step
        logs = []
        collected = []

        with TrainLoop(max_step=7, print_func=logs.append) as loop:
            for epoch in loop.iter_epochs():
                for step, batch in loop.iter_steps(data_generator(3)):
                    collected.append((epoch, step, batch))
                    loop.collect(batch=batch)
                loop.collect(test=epoch)

        assert_match_logs(logs, [
            '[Epoch 1, Step 3/7, ETA ...s] epoch_time: ...s; step_time: '
            '...s (±...s); batch: 1 (±0.816497); test: 1',
            '[Epoch 2, Step 6/7, ETA ...s] epoch_time: ...s; step_time: '
            '...s (±...s); batch: 1 (±0.816497); test: 2',
            '[Epoch 3, Step 7/7, ETA ...s] epoch_time: ...s; step_time: '
            '...s; batch: 0; test: 3',
        ])
        self.assertListEqual(collected, [
            (1, 1, 0), (1, 2, 1), (1, 3, 2), (2, 4, 0), (2, 5, 1), (2, 6, 2),
            (3, 7, 0)
        ])

        # test iter_steps only
        logs = []
        collected = []

        with TrainLoop(print_func=logs.append,
                       log_freq='every 2 steps') as loop:
            for step, batch in loop.iter_steps(data_generator(4)):
                collected.append((loop.epoch, step, batch))
                loop.collect(batch=batch)

        assert_match_logs(logs, [
            '[Step 2] step_time: ...s (±...s); batch: 0.5 (±0.5)',
            '[Step 4] step_time: ...s (±...s); batch: 2.5 (±0.5)',
        ])
        self.assertListEqual(collected, [
            (0, 1, 0), (0, 2, 1), (0, 3, 2), (0, 4, 3),
        ])

        # test no-reentrant of the epoch loop
        with TrainLoop() as loop:
            for _ in loop.iter_epochs(count=1):
                with pytest.raises(RuntimeError,
                                   match='Another epoch loop has been opened'):
                    for _ in loop.iter_epochs(count=1):
                        pass

        # test no-reentrant of the step loop
        with TrainLoop() as loop:
            for _ in loop.iter_steps(count=1):
                with pytest.raises(RuntimeError,
                                   match='Another step loop has been opened'):
                    for _ in loop.iter_steps(count=1):
                        pass

        # test no unstoppable loop protection
        with TrainLoop() as loop:
            with pytest.raises(RuntimeError,
                               match='`data_generator` is required when '
                                     '`max_step`, `limit` and `count` are all '
                                     'None, so as to prevent an unstoppable '
                                     'loop'):
                for _ in loop.iter_steps():
                    pass

    def test_get_progress(self):
        # test max_epoch is configured but not max_step
        with TrainLoop(max_epoch=5) as loop:
            self.assertEqual(loop.get_progress(), 0.)
            for epoch in loop.iter_epochs():
                self.assertAlmostEqual(loop.get_progress(), (epoch - 1) * .2)
                for i, step in enumerate(loop.iter_steps(count=5)):
                    if epoch > 1:
                        self.assertAlmostEqual(
                            loop.get_progress(),
                            (epoch - 1) * .2 + i * .04
                        )
                    else:
                        self.assertAlmostEqual(loop.get_progress(), 0.)
                if epoch == 1:
                    self.assertAlmostEqual(
                        loop.get_progress(), (epoch - 1) * .2)
                else:
                    self.assertAlmostEqual(loop.get_progress(), epoch * .2)
            self.assertAlmostEqual(loop.get_progress(), 1.)

        # test max_step is configured but not max_epoch
        with TrainLoop(max_step=10) as loop:
            self.assertAlmostEqual(loop.get_progress(), 0.)
            for step in loop.iter_steps(count=5):
                self.assertAlmostEqual(loop.get_progress(), (step - 1) * 0.1)
            self.assertAlmostEqual(loop.get_progress(), 0.5)

            for step in loop.iter_steps(count=5):
                self.assertAlmostEqual(loop.get_progress(), (step - 1) * 0.1)
            self.assertAlmostEqual(loop.get_progress(), 1.)

    def test_collect(self):
        logs = []

        with TrainLoop(max_epoch=3, print_func=logs.append) as loop:
            for _ in loop.iter_epochs():
                for step, batch in loop.iter_steps([1, 2, 3]):
                    # note kwargs should override the dict
                    loop.collect({'batch': -1}, batch=batch)
                    with loop.timeit('test_time'):
                        pass
                with loop.collector('test_acc') as c:
                    c.collect(1., weight=2.)
                    c.collect(np.array([2., 3.]))

        assert_match_logs(logs, [
            '[Epoch 1/3, Step 3, ETA ...s] epoch_time: ...s; step_time: ...s '
            '(±...s); test_time: ...s (±...s); batch: 2 (±0.816497); '
            'test_acc: 1.75',
            '[Epoch 2/3, Step 6, ETA ...s] epoch_time: ...s; step_time: ...s '
            '(±...s); test_time: ...s (±...s); batch: 2 (±0.816497); '
            'test_acc: 1.75',
            '[Epoch 3/3, Step 9, ETA ...s] epoch_time: ...s; step_time: ...s '
            '(±...s); test_time: ...s (±...s); batch: 2 (±0.816497); '
            'test_acc: 1.75',
        ])

        # test empty collector
        logs = []

        with TrainLoop(max_epoch=1, print_func=logs.append) as loop:
            for _ in loop.iter_epochs():
                for step, batch in loop.iter_steps([1, 2, 3]):
                    loop.collect({'batch': batch})
                with loop.collector('test_acc'):
                    pass

        assert_match_logs(logs, [
            '[Step 3, ETA ...s] epoch_time: ...s; step_time: ...s (±...s); '
            'batch: 2 (±0.816497)'
        ])

        # test errors
        with TrainLoop(max_epoch=1) as loop:
            for _ in loop.iter_epochs():
                with pytest.raises(TypeError,
                                   match='`metrics` should be a dict'):
                    loop.collect(object())

        with TrainLoop(max_epoch=1) as loop:
            for _ in loop.iter_epochs():
                with pytest.raises(ValueError,
                                   match='`metric_name` does not end with '
                                         '"_time": \'abc\''):
                    with loop.timeit('abc'):
                        pass

    def test_checkpoint(self):
        with TemporaryDirectory() as temp_dir:
            # test manual save
            a = SimpleStatefulObject()
            b = SimpleStatefulObject()
            objects = {'a': a, 'b': b}

            with TrainLoop(objects=objects, checkpoint_root=temp_dir) as loop:
                a.value = 12303
                b.value = 45603
                loop.memo.note = 'hello'
                loop.state.epoch = 1
                loop.state.step = 3
                loop.make_checkpoint()

            self.assertTrue(os.path.isdir(os.path.join(temp_dir, '3')))

            # test load and auto-save
            logs = []
            collected = []

            a = SimpleStatefulObject()
            b = SimpleStatefulObject()
            objects = {'a': a, 'b': b}

            with TrainLoop(objects=objects,
                           max_epoch=4,
                           print_func=logs.append,
                           checkpoint_root=temp_dir,
                           checkpoint_freq='every epoch',
                           max_checkpoint_to_keep=3) as loop:
                self.assertEqual(a.value, 12303)
                self.assertEqual(b.value, 45603)
                self.assertEqual(loop.memo.note, 'hello')
                self.assertEqual(loop.epoch, 1)
                self.assertEqual(loop.step, 3)

                for epoch in loop.iter_epochs():
                    for step, batch in loop.iter_steps([0, 1, 2]):
                        loop.memo.note = f'hello{step}'
                        a.value = 12300 + step
                        b.value = 45600 + step
                        collected.append((epoch, step, batch))
                        loop.collect(batch=batch)

            self.assertFalse(os.path.isdir(os.path.join(temp_dir, '3')))
            self.assertTrue(os.path.isdir(os.path.join(temp_dir, '6')))
            self.assertTrue(os.path.isdir(os.path.join(temp_dir, '9')))
            self.assertTrue(os.path.isdir(os.path.join(temp_dir, '12')))

            assert_match_logs(logs, [
                'Resume training: epoch 1, step 3, from checkpoint '
                'directory \'...3\'',
                '[Epoch 2/4, Step 6, ETA ...s] epoch_time: ...s; '
                'step_time: ...s (±...s); batch: 1 (±0.816497)',
                '[Epoch 3/4, Step 9, ETA ...s] epoch_time: ...s; '
                'step_time: ...s (±...s); batch: 1 (±0.816497)',
                '[Epoch 4/4, Step 12, ETA ...s] epoch_time: ...s; '
                'step_time: ...s (±...s); batch: 1 (±0.816497)'
            ])
            self.assertListEqual(collected, [
                (2, 4, 0), (2, 5, 1), (2, 6, 2), (3, 7, 0), (3, 8, 1),
                (3, 9, 2), (4, 10, 0), (4, 11, 1), (4, 12, 2),
            ])

            # test load from latest
            a = SimpleStatefulObject()
            b = SimpleStatefulObject()
            objects = {'a': a, 'b': b}

            with TrainLoop(objects=objects,
                           checkpoint_root=temp_dir,
                           restore_checkpoint=True) as loop:
                self.assertEqual(a.value, 12312)
                self.assertEqual(b.value, 45612)
                self.assertEqual(loop.memo.note, 'hello12')
                self.assertEqual(loop.epoch, 4)
                self.assertEqual(loop.step, 12)

            # test load from user specified checkpoint
            a = SimpleStatefulObject()
            b = SimpleStatefulObject()
            objects = {'a': a, 'b': b}

            with TrainLoop(objects=objects,
                           checkpoint_root=temp_dir,
                           restore_checkpoint=os.path.join(temp_dir, '9')
                           ) as loop:
                self.assertEqual(a.value, 12309)
                self.assertEqual(b.value, 45609)
                self.assertEqual(loop.memo.note, 'hello9')
                self.assertEqual(loop.epoch, 3)
                self.assertEqual(loop.step, 9)

            # test not load checkpoint
            a = SimpleStatefulObject()
            b = SimpleStatefulObject()
            objects = {'a': a, 'b': b}

            with TrainLoop(objects=objects,
                           checkpoint_root=temp_dir,
                           restore_checkpoint=False
                           ) as loop:
                self.assertFalse(hasattr(a, 'value'))
                self.assertFalse(hasattr(b, 'value'))
                self.assertFalse(hasattr(loop.memo, 'note'))
                self.assertEqual(loop.epoch, 0)
                self.assertEqual(loop.step, 0)

        with TemporaryDirectory() as temp_dir:
            # test auto-save checkpoint by steps
            with TrainLoop(max_step=10,
                           checkpoint_root=temp_dir,
                           checkpoint_freq='every 2 steps',
                           max_checkpoint_to_keep=3) as loop:
                for step, batch in loop.iter_steps(range(10)):
                    loop.memo.value = (step, batch)

                self.assertFalse(os.path.isdir(os.path.join(temp_dir, '2')))
                self.assertFalse(os.path.isdir(os.path.join(temp_dir, '4')))
                self.assertTrue(os.path.isdir(os.path.join(temp_dir, '6')))
                self.assertTrue(os.path.isdir(os.path.join(temp_dir, '8')))
                self.assertTrue(os.path.isdir(os.path.join(temp_dir, '10')))

            with TrainLoop(max_step=10,
                           checkpoint_root=temp_dir) as loop:
                self.assertTupleEqual(loop.memo.value, (10, 9))

    def test_events(self):
        logs = []
        p = logs.append
        loop = TrainLoop(max_epoch=5, print_func=logs.append)

        loop.on_enter_loop.do(lambda: p('enter loop'))
        loop.on_exit_loop.do(lambda: p('exit loop'))
        loop.on_enter_epoch.do(lambda e: p(f'enter epoch {e}'))
        loop.on_exit_epoch.do(lambda e: p(f'exit epoch {e}'))
        loop.on_enter_step.do(
            lambda s, batch_data: p(f'enter step {s}, {batch_data}'))
        loop.on_exit_step.do(lambda s: p(f'exit step {s}'))
        loop.on_metrics_collected.do(
            lambda metrics: p(f'metrics collected {metrics}'))
        loop.on_stats_printed.do(lambda stats: p(f'stats printed {stats}'))

        loop.do_after(lambda: p('after every 2 epochs'), 'every 2 epochs')
        loop.do_after(lambda: p('after every 3 steps'), 'every 3 steps')
        loop.do_after_epochs(lambda: p('after every 3 epochs'), epochs=3)
        loop.do_after_steps(lambda: p('after every 2 steps'), steps=2)

        with loop:
            p('check 1')
            for epoch in loop.iter_epochs():
                p(f'check 2: {epoch}')
                for step, batch in loop.iter_steps(range(3)):
                    p(f'check 3: {step} {batch}')
                    loop.collect(batch=batch)
                p(f'check 4: {epoch}')
                loop.collect(test=epoch)
            p('check 5')

        assert_match_logs(logs, [
            'enter loop',
            'check 1',
            'enter epoch 1',
            'check 2: 1',
            'enter step 1, 0',
            'check 3: 1 0',
            "metrics collected {'batch': 0}",
            'exit step 1',
            "metrics collected {'step_time': ...}",
            'enter step 2, 1',
            'check 3: 2 1',
            "metrics collected {'batch': 1}",
            'exit step 2',
            'after every 2 steps',
            "metrics collected {'step_time': ...}",
            'enter step 3, 2',
            'check 3: 3 2',
            "metrics collected {'batch': 2}",
            'exit step 3',
            'after every 3 steps',
            "metrics collected {'step_time': ...}",
            'check 4: 1',
            "metrics collected {'test': 1}",
            'exit epoch 1',
            "metrics collected {'epoch_time': ...}",
            '[Epoch 1/5, Step 3, ETA ...s] epoch_time: ...s; step_time: ... (±...s); batch: 1 (±0.816497); test: 1',
            "stats printed {'batch': (array(1.), array(0.81649658)), 'step_time': (..., ...), 'test': (array(1.), None), 'epoch_time': (..., None)}",
            'enter epoch 2',
            'check 2: 2',
            'enter step 4, 0',
            'check 3: 4 0',
            "metrics collected {'batch': 0}",
            'exit step 4',
            'after every 2 steps',
            "metrics collected {'step_time': ...}",
            'enter step 5, 1',
            'check 3: 5 1',
            "metrics collected {'batch': 1}",
            'exit step 5',
            "metrics collected {'step_time': ...}",
            'enter step 6, 2',
            'check 3: 6 2',
            "metrics collected {'batch': 2}",
            'exit step 6',
            'after every 3 steps',
            'after every 2 steps',
            "metrics collected {'step_time': ...}",
            'check 4: 2',
            "metrics collected {'test': 2}",
            'exit epoch 2',
            'after every 2 epochs',
            "metrics collected {'epoch_time': ...}",
            '[Epoch 2/5, Step 6, ETA ...s] epoch_time: ...s; step_time: ... (±...s); batch: 1 (±0.816497); test: 2',
            "stats printed {'batch': (array(1.), array(0.81649658)), 'step_time': (..., ...), 'test': (array(2.), None), 'epoch_time': (..., None)}",
            'enter epoch 3',
            'check 2: 3',
            'enter step 7, 0',
            'check 3: 7 0',
            "metrics collected {'batch': 0}",
            'exit step 7',
            "metrics collected {'step_time': ...}",
            'enter step 8, 1',
            'check 3: 8 1',
            "metrics collected {'batch': 1}",
            'exit step 8',
            'after every 2 steps',
            "metrics collected {'step_time': ...}",
            'enter step 9, 2',
            'check 3: 9 2',
            "metrics collected {'batch': 2}",
            'exit step 9',
            'after every 3 steps',
            "metrics collected {'step_time': ...}",
            'check 4: 3',
            "metrics collected {'test': 3}",
            'exit epoch 3',
            'after every 3 epochs',
            "metrics collected {'epoch_time': ...}",
            '[Epoch 3/5, Step 9, ETA ...s] epoch_time: ...s; step_time: ... (±...s); batch: 1 (±0.816497); test: 3',
            "stats printed {'batch': (array(1.), array(0.81649658)), 'step_time': (..., ...), 'test': (array(3.), None), 'epoch_time': (..., None)}",
            'enter epoch 4',
            'check 2: 4',
            'enter step 10, 0',
            'check 3: 10 0',
            "metrics collected {'batch': 0}",
            'exit step 10',
            'after every 2 steps',
            "metrics collected {'step_time': ...}",
            'enter step 11, 1',
            'check 3: 11 1',
            "metrics collected {'batch': 1}",
            'exit step 11',
            "metrics collected {'step_time': ...}",
            'enter step 12, 2',
            'check 3: 12 2',
            "metrics collected {'batch': 2}",
            'exit step 12',
            'after every 3 steps',
            'after every 2 steps',
            "metrics collected {'step_time': ...}",
            'check 4: 4',
            "metrics collected {'test': 4}",
            'exit epoch 4',
            'after every 2 epochs',
            "metrics collected {'epoch_time': ...}",
            '[Epoch 4/5, Step 12, ETA ...s] epoch_time: ...s; step_time: ... (±...s); batch: 1 (±0.816497); test: 4',
            "stats printed {'batch': (array(1.), array(0.81649658)), 'step_time': (..., ...), 'test': (array(4.), None), 'epoch_time': (..., None)}",
            'enter epoch 5',
            'check 2: 5',
            'enter step 13, 0',
            'check 3: 13 0',
            "metrics collected {'batch': 0}",
            'exit step 13',
            "metrics collected {'step_time': ...}",
            'enter step 14, 1',
            'check 3: 14 1',
            "metrics collected {'batch': 1}",
            'exit step 14',
            'after every 2 steps',
            "metrics collected {'step_time': ...}",
            'enter step 15, 2',
            'check 3: 15 2',
            "metrics collected {'batch': 2}",
            'exit step 15',
            'after every 3 steps',
            "metrics collected {'step_time': ...}",
            'check 4: 5',
            "metrics collected {'test': 5}",
            'exit epoch 5',
            "metrics collected {'epoch_time': ...}",
            '[Epoch 5/5, Step 15, ETA ...s] epoch_time: ...s; step_time: ... (±...s); batch: 1 (±0.816497); test: 5',
            "stats printed {'batch': (array(1.), array(0.81649658)), 'step_time': (..., ...), 'test': (array(5.), None), 'epoch_time': (..., None)}",
            'check 5',
            'exit loop'
        ])

        with pytest.raises(ValueError,
                           match=r'`freq` must not be TrainLoopFreq\(never\)'):
            loop.do_after(lambda: print('never'), 'never')

    def test_run(self):
        def train_fn(loop, x):
            loop.collect(mean=np.mean(x))

        stream = DataStream.int_seq(5, batch_size=3)

        # test `run()`
        logs = []
        with TrainLoop(max_epoch=3, print_func=logs.append) as loop:
            loop.run(partial(train_fn, loop), stream)

        assert_match_logs(logs, [
            '[Epoch 1/3, Step 2, ETA ...s] epoch_time: ...s; step_time: ...s '
            '(±...s); mean: 2.25 (±1.25)',
            '[Epoch 2/3, Step 4, ETA ...s] epoch_time: ...s; step_time: ...s '
            '(±...s); mean: 2.25 (±1.25)',
            '[Epoch 3/3, Step 6, ETA ...s] epoch_time: ...s; step_time: ...s '
            '(±...s); mean: 2.25 (±1.25)',
        ])

        # test `run_epochs()`
        logs = []
        with TrainLoop(max_epoch=3, print_func=logs.append) as loop:
            loop.run_epochs(partial(train_fn, loop), stream)

        assert_match_logs(logs, [
            '[Epoch 1/3, Step 2, ETA ...s] epoch_time: ...s; step_time: ...s '
            '(±...s); mean: 2.25 (±1.25)',
            '[Epoch 2/3, Step 4, ETA ...s] epoch_time: ...s; step_time: ...s '
            '(±...s); mean: 2.25 (±1.25)',
            '[Epoch 3/3, Step 6, ETA ...s] epoch_time: ...s; step_time: ...s '
            '(±...s); mean: 2.25 (±1.25)',
        ])

        # test `run_epochs(limit=2)`
        logs = []
        with TrainLoop(max_epoch=3, print_func=logs.append) as loop:
            loop.run_epochs(partial(train_fn, loop), stream, limit=2)

        assert_match_logs(logs, [
            '[Epoch 1/3, Step 2, ETA ...s] epoch_time: ...s; step_time: ...s '
            '(±...s); mean: 2.25 (±1.25)',
            '[Epoch 2/3, Step 4, ETA ...s] epoch_time: ...s; step_time: ...s '
            '(±...s); mean: 2.25 (±1.25)',
        ])

        # test `run_steps()`
        logs = []
        with TrainLoop(max_epoch=3, print_func=logs.append) as loop:
            for _ in loop.iter_epochs():
                loop.run_steps(partial(train_fn, loop), iter(stream))

        assert_match_logs(logs, [
            '[Epoch 1/3, Step 2, ETA ...s] epoch_time: ...s; step_time: ...s '
            '(±...s); mean: 2.25 (±1.25)',
            '[Epoch 2/3, Step 4, ETA ...s] epoch_time: ...s; step_time: ...s '
            '(±...s); mean: 2.25 (±1.25)',
            '[Epoch 3/3, Step 6, ETA ...s] epoch_time: ...s; step_time: ...s '
            '(±...s); mean: 2.25 (±1.25)',
        ])

        # test `run_steps(count=1)`
        logs = []
        with TrainLoop(max_epoch=3, print_func=logs.append) as loop:
            for _ in loop.iter_epochs():
                loop.run_steps(partial(train_fn, loop), iter(stream), count=1)

        assert_match_logs(logs, [
            '[Epoch 1/3, Step 1, ETA ...s] epoch_time: ...s; step_time: ...s; '
            'mean: 1',
            '[Epoch 2/3, Step 2, ETA ...s] epoch_time: ...s; step_time: ...s; '
            'mean: 1',
            '[Epoch 3/3, Step 3, ETA ...s] epoch_time: ...s; step_time: ...s; '
            'mean: 1',
        ])

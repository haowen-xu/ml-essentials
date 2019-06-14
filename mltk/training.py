import os
import pickle
import re
import shutil
import time
from contextlib import contextmanager
from functools import wraps
from logging import getLogger
from typing import *

from .data.stream import DataStream
from .events import EventHost, Event
from .logging import MetricLogger, MetricType, StatisticsCollector
from .stateful import (StatefulObject, SimpleStatefulObject,
                       StatefulObjectGroup, StateSaver)
from .utils import optional_apply, format_duration, DisposableContext, ETA

__all__ = ['TrainLoopFreq', 'TrainLoop']

TrainObjectsType = Union[Dict[str, StatefulObject], StatefulObject]
MetricsDict = Dict[str, MetricType]
MetricStatsDict = Dict[str, Tuple[MetricType, Optional[MetricType]]]
LogProcessor = Callable[[str], str]
DataGeneratorType = Union[DataStream, Iterable, Iterator]
TrainLoopFreqLiteral = Union[str, 'TrainLoopFreq']

_EPOCH_TIME_METRIC = 'epoch_time'
_STEP_TIME_METRIC = 'step_time'
_POSITIVE_INT_PATTERN = re.compile(r'^\d+$')


def is_positive_int(s) -> bool:
    return bool(_POSITIVE_INT_PATTERN.match(s))


class TrainLoopFreq(object):
    """
    Base class to represent the frequency of an event callback in
    :class:`TrainLoop`.

    >>> freq = TrainLoopFreq.parse('every epoch')
    >>> freq
    TrainLoopFreq(every epoch)
    >>> freq.epochs, freq.steps
    (1, None)

    >>> freq = TrainLoopFreq.parse('every 2 epochs')
    >>> freq
    TrainLoopFreq(every 2 epochs)
    >>> freq.epochs, freq.steps
    (2, None)

    >>> freq = TrainLoopFreq.parse('every step')
    >>> freq
    TrainLoopFreq(every step)
    >>> freq.epochs, freq.steps
    (None, 1)

    >>> freq = TrainLoopFreq.parse('every 2 steps')
    >>> freq
    TrainLoopFreq(every 2 steps)
    >>> freq.epochs, freq.steps
    (None, 2)

    >>> freq = TrainLoopFreq.parse('never')
    >>> freq
    TrainLoopFreq(never)
    >>> freq.epochs, freq.steps
    (None, None)

    >>> freq = TrainLoopFreq.parse(None)
    >>> freq
    TrainLoopFreq(never)
    """

    NEVER: 'TrainLoopFreq'
    EVERY_EPOCH: 'TrainLoopFreq'
    EVERY_STEP: 'TrainLoopFreq'

    def __init__(self, epochs: Optional[int] = None,
                 steps: Optional[int] = None):
        """
        Construct a new :class:`TrainLoopFreq`.

        Args:
            epochs: The frequency of epochs.
            steps: The frequency of steps.

        Raises:
            ValueError: If both `epochs` and `steps` are specified.

                >>> freq = TrainLoopFreq(epochs=1, steps=2)
                Traceback (most recent call last):
                    ...
                ValueError: only one of `epochs`, `steps` can be specified
        """
        if epochs is not None and steps is not None:
            raise ValueError('only one of `epochs`, `steps` can be specified')
        if epochs is not None:
            epochs = int(epochs)
        if steps is not None:
            steps = int(steps)

        self._epochs = epochs
        self._steps = steps

    def __repr__(self):
        if self.epochs is not None:
            if self.epochs == 1:
                s = 'every epoch'
            else:
                s = f'every {self.epochs} epochs'
        elif self.steps is not None:
            if self.steps == 1:
                s = 'every step'
            else:
                s = f'every {self.steps} steps'
        else:
            s = 'never'
        return f'TrainLoopFreq({s})'

    def __eq__(self, other):
        return (
            isinstance(other, TrainLoopFreq) and
            other.epochs == self.epochs and
            other.steps == self.steps
        )

    def __hash__(self):
        return hash((self.epochs, self.steps))

    @property
    def epochs(self) -> Optional[int]:
        """Get the epoch frequency."""
        return self._epochs

    @property
    def steps(self) -> Optional[int]:
        """Get the step frequency."""
        return self._steps

    FREQ_PATTERN = re.compile(r'^(?:every|per)\s+(\d+)\s+(step|epoch)s?$')
    UNIT_FREQ_PATTERN = re.compile(r'^(?:every|per)\s+(step|epoch)$')

    @classmethod
    def parse(cls,
              freq: Optional[TrainLoopFreqLiteral]
              ) -> 'TrainLoopFreq':
        """
        Parse `freq` into a :class:`TrainLoopFreq`.

        Args:
            freq: A None, a str representing train loop frequency, or a
                :class:`TrainLoopFreq` object.

        Returns:
            The parsed :class:`TrainLoopFreq` object.

        Raises:
            ValueError: If `freq` is a str, but cannot be parsed as
                a frequency.

                >>> TrainLoopFreq.parse('invalid freq')
                Traceback (most recent call last):
                    ...
                ValueError: cannot parse `freq`: 'invalid freq'
        """
        if freq is None:
            return TrainLoopFreq(epochs=None, steps=None)

        elif isinstance(freq, str):
            freq = freq.lower()

            # try pattern 'never'
            if freq == 'never':
                return TrainLoopFreq()

            # try pattern '(every|step) \d+ (epoch|step)s?'
            m = cls.FREQ_PATTERN.match(freq)
            if m:
                freq, unit = m.groups()
                freq = int(freq)
                if unit.lower() == 'epoch':
                    return TrainLoopFreq(epochs=freq)
                else:
                    return TrainLoopFreq(steps=freq)

            # try pattern '(every|step) (epoch|step)'
            m = cls.UNIT_FREQ_PATTERN.match(freq)
            if m:
                unit = m.group(1)
                if unit.lower() == 'epoch':
                    return TrainLoopFreq(epochs=1)
                else:
                    return TrainLoopFreq(steps=1)

            raise ValueError(f'cannot parse `freq`: {freq!r}')

        elif isinstance(freq, TrainLoopFreq):
            return freq

        else:
            raise TypeError(f'`freq` is neither a str nor a TrainLoopFreq: '
                            f'{freq!r}')

    def is_epoch_matches(self, epoch: int) -> bool:
        """
        Test whether or not the specified `epoch` matches the frequency
        represented by this :class:`TrainLoopFreq`.

        >>> freq = TrainLoopFreq(epochs=2)
        >>> freq.is_epoch_matches(2)
        True
        >>> freq.is_epoch_matches(3)
        False
        >>> freq.is_epoch_matches(4)
        True

        >>> freq = TrainLoopFreq(steps=2)
        >>> freq.is_epoch_matches(2)
        False

        >>> freq = TrainLoopFreq()
        >>> freq.is_epoch_matches(1)
        False

        Args:
            epoch: The epoch number.
        """
        return self.epochs is not None and epoch % self.epochs == 0

    def is_step_matches(self, step: int) -> bool:
        """
        Test whether or not the specified `step` matches the frequency
        represented by this :class:`TrainLoopFreq`.

        >>> freq = TrainLoopFreq(steps=2)
        >>> freq.is_step_matches(2)
        True
        >>> freq.is_step_matches(3)
        False
        >>> freq.is_step_matches(4)
        True

        >>> freq = TrainLoopFreq(epochs=2)
        >>> freq.is_step_matches(2)
        False

        >>> freq = TrainLoopFreq()
        >>> freq.is_step_matches(1)
        False

        Args:
            step: The step number.
        """
        return self.steps is not None and step % self.steps == 0

    def register_callback(self, loop: 'TrainLoop', cb: Callable[..., None]):
        """
        Bind an callback `cb` to `loop.on_exit_epoch` or `loop.on_exit_step`,
        according to the frequency represented by this :class:`TrainLoopFreq`
        object.

        Args:
            loop: The train loop object.
            cb: The callback function.
        """
        # determine which event to bind, and wrap the callback
        if self.epochs is None and self.steps is None:
            return

        elif self.epochs is not None:
            @wraps(cb)
            def wrapped(*args, **kwargs):
                if loop.epoch % self.epochs == 0:
                    return cb(*args, **kwargs)
            event = loop.on_exit_epoch

        elif self.steps is not None:
            @wraps(cb)
            def wrapped(*args, **kwargs):
                if loop.step % self.steps == 0:
                    return cb(*args, **kwargs)
            event = loop.on_exit_step

        else:  # pragma: no cover
            raise RuntimeError('should not touch this branch')

        # do not use wrapper if frequency is 1
        if self.epochs == 1 or self.steps == 1:
            wrapped = cb

        # now register the event callback
        event.do(wrapped)


TrainLoopFreq.NEVER = TrainLoopFreq()
"""A :class:`TrainLoopFreq` instance which represents `never`."""

TrainLoopFreq.EVERY_EPOCH = TrainLoopFreq(epochs=1)
"""A :class:`TrainLoopFreq` instance which represents `every epoch`."""

TrainLoopFreq.EVERY_STEP = TrainLoopFreq(steps=1)
"""A :class:`TrainLoopFreq` instance which represents `every step`."""


class TrainLoopState(StatefulObject):
    """
    Internal state of a :class:`TrainLoop`, which can be saved via a
    :class:`StateSaver`.

    >>> state = TrainLoopState()
    >>> state
    TrainLoopState(epoch=0, step=0)
    >>> state.epoch = 1
    >>> state.step += 2
    >>> state
    TrainLoopState(epoch=1, step=2)
    >>> state.get_state_dict()
    {'epoch': 1, 'step': 2}
    >>> state.set_state_dict({'epoch': 3, 'step': 4})
    >>> state
    TrainLoopState(epoch=3, step=4)
    """

    def __init__(self, epoch: int = 0, step: int = 0):
        self._epoch = epoch
        self._step = step

    def __repr__(self):
        return f'TrainLoopState(epoch={self.epoch}, step={self.step})'

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'step': self.step,
        }

    def set_state_dict(self, state: Dict[str, Any]):
        self.__dict__.update({
            '_epoch': int(state['epoch']),
            '_step': int(state['step']),
        })


class TrainLoopCheckpointManager(object):
    """Class to manage checkpoints of a train loop."""

    STATE_FILE_NAME = 'state.npz'
    OBJECTS_FILE_NAME = 'objects.npz'

    def __init__(self,
                 root_dir: str,
                 state: TrainLoopState,
                 memo: StatefulObject,
                 objects: Optional[TrainObjectsType] = None,
                 max_to_keep: Optional[int] = None,
                 pickle_protocol=pickle.HIGHEST_PROTOCOL):
        """
        Construct a new :class:`TrainLoopCheckpointManager`.

        Args:
            root_dir: The root directory of the checkpoint dirs.
                Each checkpoint dir should be f'{root_dir}/{loop.step}'.
            state: The train loop state object.
            memo: The train loop memo object.
            objects: User specified objects to save in the checkpoints.
            max_to_keep: Maximum number of historical checkpoints to keep.
                If :obj:`None`, keep infinite checkpoints.
            pickle_protocol: The pickle protocol to use.
        """

        # prepare for the state objects
        if max_to_keep is not None:
            max_to_keep = int(max_to_keep)
            if max_to_keep < 1:
                raise ValueError(f'`max_to_keep` must be at least 1: '
                                 f'got {max_to_keep}')

        self._root_dir = root_dir
        self._state = state
        self._state_obj = StatefulObjectGroup({
            'state': state,
            'memo': memo,
        })
        self._objects = objects
        self._max_to_keep = max_to_keep

        # state savers
        self._state_saver = StateSaver(self._state_obj, pickle_protocol)
        if objects is not None:
            self._objects_saver = StateSaver(self._objects, pickle_protocol)
        else:
            self._objects_saver = None  # type: Optional[StateSaver]

        # get existing checkpoint dirs
        checkpoint_steps = []
        if os.path.isdir(self._root_dir):
            for name in os.listdir(self._root_dir):
                path = os.path.join(self._root_dir, name)
                if is_positive_int(name) and os.path.isdir(path):
                    checkpoint_steps.append(int(name))
        checkpoint_steps.sort()
        self._checkpoint_dirs = [os.path.join(self._root_dir, str(step))
                                 for step in checkpoint_steps]

    def save(self) -> str:
        """
        Save a checkpoint.

        The checkpoint directory will be f'{root_dir}/{loop.step}'.
        """
        step = self._state.step
        checkpoint_dir = os.path.join(self._root_dir, str(step))
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        try:
            # save the state and objects
            state_file = os.path.join(checkpoint_dir, self.STATE_FILE_NAME)
            self._state_saver.save(state_file)

            if self._objects_saver is not None:
                objects_file = os.path.join(
                    checkpoint_dir, self.OBJECTS_FILE_NAME)
                self._objects_saver.save(objects_file)

            self._checkpoint_dirs.append(checkpoint_dir)

            # cleanup old checkpoints
            if self._max_to_keep is not None:
                checkpoint_dirs = self._checkpoint_dirs
                dirs_to_delete = checkpoint_dirs[: -self._max_to_keep]
                self._checkpoint_dirs = checkpoint_dirs[-self._max_to_keep:]
                for path in dirs_to_delete:
                    try:
                        shutil.rmtree(path)
                    except Exception:  # pragma: no cover
                        getLogger(__name__).warning(
                            'Failed to delete old checkpoint dir: %s',
                            path, exc_info=True
                        )
        except:
            shutil.rmtree(checkpoint_dir)
            raise
        else:
            return checkpoint_dir

    def latest_checkpoint_dir(self) -> Optional[str]:
        """
        Get the latest checkpoint directory, or :obj:`None` if no checkpoint
        has been saved.
        """
        if self._checkpoint_dirs:
            return self._checkpoint_dirs[-1]

    def restore(self, checkpoint_dir):
        """
        Restore a checkpoint.

        Args:
            checkpoint_dir: The checkpoint directory.
        """
        if not os.path.isdir(checkpoint_dir):
            raise IOError(f'`checkpoint_dir` not exist: {checkpoint_dir}')

        state_file = os.path.join(checkpoint_dir, self.STATE_FILE_NAME)
        old_state = self._state_obj.get_state_dict()
        try:
            self._state_saver.load(state_file)
            if self._objects_saver is not None:
                objects_file = os.path.join(
                    checkpoint_dir, self.OBJECTS_FILE_NAME)
                self._objects_saver.load(objects_file)
        except:
            self._state_obj.set_state_dict(old_state)
            raise


class TrainLoop(DisposableContext):
    """
    Train loop object.

    This class provides a set of convenient methods to write a train loop.
    It can collect various training metrics, estimating training time ahead,
    persist training state, and generating formatted training logs.
    Furthermore, it fires a series of events, such that it is easy to extend
    a train loop object.

    There are multiple usage patterns of :class:`TrainLoop`, listed as follows.

    Basic Usage
    ===========

    :class:`TrainLoop` can be used to generate epoch and mini-batch iterators.
    You may write any train and evaluation code within the loops.  As long as
    you submit the training and evaluation metrics via
    :meth:`collect()`, the :class:`TrainLoop` object will produce
    the train logs automatically, which includes the statistics of the metrics
    and time consumption.  For example::

        train_stream = DataStream.arrays([train_x, train_y], batch_size=64,
                                        shuffle=True, skip_incomplete=True)
        test_stream = DataStream.arrays([test_x, test_y], batch_size=256)

        with TrainLoop(max_epoch=100) as loop:
            for epoch in loop.iter_epochs():
                for step, [x, y] in loop.iter_steps(train_stream):
                    # run train step, and compute `train_loss`.
                    train_loss = ...

                    # `train_loss` can either be a scalar (the average train
                    # loss of the batch), or a 1-d array (the element-wise
                    # train loss of each train data).
                    loop.collect(train_loss=train_loss)

                if epoch % 100 == 0:
                    # every 100 epochs, run evaluation to get `test_acc`.
                    # we use :meth:`collector()` to construct a
                    # :class:`StatisticsCollector`, in order to collect
                    # all the accuracies from each test mini-batch.
                    with loop.collector('test_acc') as c, \
                            loop.timeit('test_time'):
                        for [x, y] in test_stream:
                            # run test step, and compute `test_acc`.
                            test_acc = ...

                            # suppose `test_acc` is a scalar, the average
                            # accuracy of the test batch:
                            c.collect(test_acc, weight=len(x))

                            # if `test_acc` is a 1-d array, indicating the
                            # element-wise accuracy of each test data, you
                            # should discard the `weight` argument as:
                            c.collect(test_acc)

    Sometimes you may write the train and evaluation code as functions.
    In such case, you can make use of the train loop events.  For example::

        def train_fn():
            train_loss = ...
            loop.collect(train_loss=train_loss)

        def evaluate_fn():
            with loop.collector('test_acc') as c, \
                    loop.timeit('test_time'):
                for [x, y] in test_stream:
                    test_acc = ...
                    c.collect(test_acc, weight=len(x))

        with TrainLoop(max_epoch=100) as loop:
            # registers a callback, which will be run every 100 epochs.
            # you may also consider `do_after_steps()` or `do_after()`.
            loop.do_after_epochs(evaluate_fn, epochs=100)

            for epoch in loop.iter_epochs():
                for step, [x, y] in loop.iter_steps(train_stream):
                    train_fn()

                # you do not need to call `evaluate_fn()`.  loop will call it.

    The above code can be further simplified by using :meth:`run()`::

        with TrainLoop(max_epoch=100) as loop:
            loop.do_after_epochs(evaluate_fn, epochs=100)

            # use `run()` to finish all training epochs and steps.
            loop.run(train_fn, train_stream)

    Or you may use :meth:`run_epochs()` and :meth:`run_steps()` to achieve
    fine-grained control over the loop::

        with TrainLoop(max_epoch=100) as loop:
            loop.do_after_epochs(evaluate_fn, epochs=100)

            # run the first 10 epochs by `run_epochs()`.
            loop.run_epochs(train_fn, train_stream, limit=10)

            # then run the remaining epochs by `run_steps()`.
            for epoch in loop.iter_epochs():  # epoch will start from 11
                loop.run_steps(train_fn, train_stream)
    """

    def __init__(self,
                 objects: Optional[TrainObjectsType] = None,
                 max_epoch: Optional[int] = None,
                 max_step: Optional[int] = None,
                 log_freq: TrainLoopFreqLiteral = TrainLoopFreq.EVERY_EPOCH,
                 checkpoint_root: Optional[str] = None,
                 checkpoint_freq: TrainLoopFreqLiteral = TrainLoopFreq.NEVER,
                 restore_checkpoint: Union[bool, str] = True,
                 max_checkpoint_to_keep: Optional[int] = None,
                 print_func: Callable[[str], None] = print):
        """
        Construct a new :class:`TrainLoop`.

        Args:
            objects: User specified objects to be saved along with train loop
                state in the checkpoints.
            max_epoch: The maximum number of epochs to run.
            max_step: The maximum number of steps to run.
            log_freq: The frequency to print logs.  Defaults to "every epoch".
            checkpoint_root: The root directory to store the checkpoints.
            checkpoint_freq: The frequency to save checkpoints.
                Defaults to "never".
            restore_checkpoint: If a :type:`bool`, indicate whether or not to
                restore from the latest checkpoint directory (if exists) in
                `checkpoint_root`.

                If a :type:`str`, it should represent an existing checkpoint
                directory, and will restore from this checkpoint.
            max_checkpoint_to_keep: Maximum number of checkpoints to keep.
                If :obj:`None`, keep infinite checkpoints.
            print_func: The function to print a log line.  Defaults to `print`.
        """

        if max_epoch is not None:
            max_epoch = int(max_epoch)
        if max_step is not None:
            max_step = int(max_step)
        log_freq = TrainLoopFreq.parse(log_freq)
        checkpoint_freq = TrainLoopFreq.parse(checkpoint_freq)

        # states and configurations
        self._objects = objects
        self._state = TrainLoopState()
        self._memo = SimpleStatefulObject()
        self._max_epoch = max_epoch
        self._max_step = max_step
        self._print_func = print_func
        self._log_freq = log_freq

        # checkpoint manager
        self._checkpoint_root = checkpoint_root
        self._checkpoint_freq = checkpoint_freq
        self._restore_checkpoint = restore_checkpoint

        if checkpoint_root is not None:
            self._checkpoint_manager = TrainLoopCheckpointManager(
                root_dir=checkpoint_root,
                state=self.state,
                memo=self.memo,
                objects=self.objects,
                max_to_keep=max_checkpoint_to_keep,
            )
        else:
            self._checkpoint_manager = None  # type: Optional[TrainLoopCheckpointManager]

        # euphemeral states
        self._eta = ETA()
        self._epoch_logger = MetricLogger()
        self._step_logger = MetricLogger()
        self._within_epoch = False
        self._within_step = False
        self._steps_per_epoch = None  # average steps per epoch
        self._epoch_start_time = None  # start time of the current epoch
        self._step_start_time = None  # start time of the current step

        # events
        self._events = EventHost()
        self._on_enter_loop = self.events['enter_loop']
        self._on_exit_loop = self.events['exit_loop']
        self._on_enter_epoch = self.events['enter_epoch']
        self._on_exit_epoch = self.events['exit_epoch']
        self._on_enter_step = self.events['enter_step']
        self._on_exit_step = self.events['exit_step']
        self._on_metrics_collected = self.events['metrics_collected']
        self._on_stats_printed = self.events['stats_printed']

    @property
    def objects(self) -> Optional[TrainObjectsType]:
        """User specified objects to save in the checkpoints."""
        return self._objects

    @property
    def state(self) -> TrainLoopState:
        """The train loop state."""
        return self._state

    @property
    def memo(self) -> StatefulObject:
        """
        The memo object.

        This object will be saved along with the train loop state in the
        checkpoints.  Users can store anything in this memo object, e.g.::

            with TrainLoop(...) as loop:
                # Test whether or not an attribute exists.  If it does not
                # exist, assign a value to it.
                # It will indeed exist if `loop` is recovered from a checkpoint.
                if not hasattr(loop.memo, 'my_state'):
                    loop.memo.my_state = 1

                # Now we can use this state
                for epoch in loop.iter_epochs():
                    loop.memo.my_state += 1
        """
        return self._memo

    @property
    def epoch(self) -> int:
        """Get the epoch counter."""
        return self.state.epoch

    @property
    def step(self) -> int:
        """Get the step counter."""
        return self.state.step

    @property
    def max_epoch(self) -> Optional[int]:
        """Get the maximum number of epochs to run."""
        return self._max_epoch

    @max_epoch.setter
    def max_epoch(self, value):
        self._max_epoch = optional_apply(int, value)

    @property
    def max_step(self) -> Optional[int]:
        """Get the maximum number of steps to run."""
        return self._max_step

    @max_step.setter
    def max_step(self, value):
        self._max_step = optional_apply(int, value)

    @property
    def log_freq(self) -> TrainLoopFreq:
        """Get the frequency to print logs."""
        return self._log_freq

    @property
    def checkpoint_freq(self) -> TrainLoopFreq:
        """Get the frequency to save checkpoints."""
        return self._checkpoint_freq

    @property
    def checkpoint_root(self) -> Optional[str]:
        """Get the root directory of the checkpoints."""
        return self._checkpoint_root

    @property
    def events(self) -> EventHost:
        """Get the event host of this train loop object."""
        return self._events

    @property
    def on_enter_loop(self) -> Event[Callable[[], None]]:
        """
        Event of entering the loop.

        Callback function type: `() -> None`

        This event will be triggered when the train loop object context
        is entered, i.e., entering the `with` clause::

            loop = TrainLoop(...)

            with loop:
                # after the above `with` clause, and right before the next
                # statement, the `on_enter_loop` will be triggered.
                ...
        """
        return self._on_enter_loop

    @property
    def on_exit_loop(self) -> Event[Callable[[], None]]:
        """
        Event of exiting the loop.

        Callback function type: `() -> None`

        This event will be triggered when the train loop object context
        is exited, i.e., exiting the `with` clause::

            loop = TrainLoop(...)

            with loop:
                ...

                # after the above statement, and before exiting the context,
                # `on_exit_loop` will be triggered.
        """
        return self._on_exit_loop

    @property
    def on_enter_epoch(self) -> Event[Callable[[int], None]]:
        """
        Event of entering an epoch.

        Callback function type: `(epoch: int) -> None`

        This event will be triggered when an epoch is entered::

            with TrainLoop(...) as loop:
                for epoch in loop.iter_epochs():
                    # each time before the following statement is executed,
                    # `on_enter_epoch` will be triggered.
                    ...
        """
        return self._on_enter_epoch

    @property
    def on_exit_epoch(self) -> Event[Callable[[int], None]]:
        """
        Event of exiting an epoch.

        Callback function type: `(epoch: int) -> None`

        This event will be triggered when an epoch is exited::

            with TrainLoop(...) as loop:
                for epoch in loop.iter_epochs():
                    ...

                    # each time after the above statement is executed,
                    # `on_exit_epoch` will be triggered.
        """
        return self._on_exit_epoch

    @property
    def on_enter_step(self) -> Event[Callable[[int, Optional[Any]], None]]:
        """
        Event of entering a step.

        Callback function type: `(step: int, batch_data: Optional[Any]) -> None`

        This event will be triggered when a step is entered::

            with TrainLoop(...) as loop:
                for epoch in loop.iter_epochs():
                    for step, [...] in loop.iter_steps(...):
                        # each time before the following statement is executed,
                        # `on_enter_step` will be triggered.
                        ...
        """
        return self._on_enter_step

    @property
    def on_exit_step(self) -> Event[Callable[[int], None]]:
        """
        Event of exiting a step.

        Callback function type: `(step: int) -> None`

        This event will be triggered when a step is exited::

            with TrainLoop(...) as loop:
                for epoch in loop.iter_epochs():
                    for step, [...] in loop.iter_steps(...):
                        ...

                        # each time after the above statement is executed,
                        # `on_exit_step` will be triggered.
        """
        return self._on_exit_step

    @property
    def on_metrics_collected(self) -> Event[Callable[[MetricsDict], None]]:
        """
        Event of training metrics having been collected.

        Callback function type: `(metrics: Dict[str, MetricType]) -> None`

        This event will be triggered each time when any new metric has been
        collected::

            with TrainLoop(...) as loop:
                for epoch in loop.iter_epochs():
                    for step, [...] in loop.iter_steps(...):
                        train_loss = ...

                        # `on_metrics_collected` will be triggered once
                        # the following statement is executed.
                        loop.collect(train_loss=train_loss)
        """
        return self._on_metrics_collected

    @property
    def on_stats_printed(self) -> Event[Callable[[MetricStatsDict], None]]:
        """
        Event of training metric statistics having been printed.

        Callback function type:
        `(stats: Dict[str, Tuple[MetricType, Optional[MetricType]]) -> None`,
        where each value is a tuple of `(mean, stddev)` statistics of the
        corresponding metric.

        This event will be triggered each time when the metric statistics
        have been printed.
        """
        return self._on_stats_printed

    def do_after(self,
                 cb: Callable[[], None],
                 freq: TrainLoopFreqLiteral):
        """
        Register a callback to run after epochs or steps.

        This is basically a convenient wrapper upon `on_exit_epoch` and
        `on_exit_step`.  However, unlike `on_exit_epoch` or `on_exit_step`,
        the callback registered by this method does not receive any argument.

        Usage::

            def my_callback():
                print('2 epochs done')

            loop.do_after(my_callback, 'every 2 epochs')

        Args:
            cb: The callback, a function without argument.
            freq: The execution frequency, e.g., "every epoch", "every 2 steps".
        """
        freq = TrainLoopFreq.parse(freq)
        if freq == TrainLoopFreq.NEVER:
            raise ValueError(f'`freq` must not be {TrainLoopFreq.NEVER}')
        freq.register_callback(self, lambda *args: cb())

    def do_after_epochs(self, cb: Callable[..., None], epochs: int = 1):
        """
        Register a callback to run after epochs.

        This is basically a convenient wrapper upon `on_exit_epoch`.
        However, unlike `on_exit_epoch`, the callback registered by this
        method does not receive any argument.

        Usage::

            def my_callback():
                print('2 epochs done')

            loop.do_after_epochs(my_callback, epochs=2)

        Args:
            cb: The callback, a function without argument.
            epochs: The execution frequency.
        """
        TrainLoopFreq(epochs=epochs).register_callback(self, lambda *args: cb())

    def do_after_steps(self, cb: Callable[..., None], steps: int = 1):
        """
        Register a callback to run after steps.

        This is basically a convenient wrapper upon `on_exit_step`.
        However, unlike `on_exit_step`, the callback registered by this
        method does not receive any argument.

        Usage::

            def my_callback():
                print('2 steps done')

            loop.do_after_steps(my_callback, steps=2)

        Args:
            cb: The callback, a function without argument.
            steps: The execution frequency.
        """
        TrainLoopFreq(steps=steps).register_callback(self, lambda *args: cb())

    def get_progress(self) -> Optional[float]:
        """
        Get the progress of training.

        Returns:
            The progress in range ``[0, 1]``, or :obj:`None` if the progress
                cannot be estimated.
        """
        max_step = self.max_step
        if max_step is None and self.max_epoch is not None and \
                self._steps_per_epoch is not None:
            max_step = self.max_epoch * self._steps_per_epoch

        if max_step:
            if self._within_step and self._step_start_time is not None:
                # _step_start_time != None, indicating the step not finished
                return (self.step - 1.) / max_step
            else:
                return float(self.step) / max_step
        elif self.max_epoch is not None:
            if self._within_epoch and self._epoch_start_time is not None:
                # _epoch_start_time != None, indicating the epoch not finished
                return (self.epoch - 1.) / self.max_epoch
            else:
                return float(self.epoch) / self.max_epoch

    def get_eta(self) -> Optional[float]:
        """
        Get the estimated time ahead (ETA).

        Returns:
            The estimated time ahead in seconds, or :obj:`None` if ETA
                cannot be esimated.
        """
        progress = self.get_progress()
        if progress is not None:
            return self._eta.get_eta(progress)

    def print(self, message, status_tag: bool = True):
        """
        Print a message.

        Args:
            message: The message text.
            status_tag: Whether or not to prepend the status tag in front of
                `message`, i.e., ``[Epoch ..., Step ..., ETA ...]``.
        """
        if status_tag and (self._within_epoch or self._within_step):
            def format_counter(value, max_value, name: str):
                if max_value is not None:
                    return f'{name} {value}/{max_value}'
                else:
                    return f'{name} {value}'

            tag = []

            # epoch counter
            if self.max_epoch != 1 and self.epoch != 0:
                tag.append(format_counter(self.epoch, self.max_epoch, 'Epoch'))

            # step counter
            tag.append(format_counter(self.step, self.max_step, 'Step'))

            # ETA
            eta = self.get_eta()
            if eta is not None:
                tag.append(f'ETA {format_duration(eta)}')

            # the final composed message
            message = f'[{", ".join(tag)}] {message}'

        self._print_func(message)

    def collect(self,
                metrics: Dict[str, MetricType] = None,
                **kwargs: MetricType):
        """
        Collect the metrics into this train loop object.

        Can be called only when an epoch or a step loop has been opened.

        Args:
            metrics: The metrics dict.
            \\**kwargs: Other named metrics.
                If a metric appears in both `metrics` and `kwargs`, the values
                in `kwargs` will override the values in `metrics`.
        """
        self._require_context()

        # merge metrics from the two arguments
        if metrics is None:
            metrics = {}
        elif metrics is not None and not isinstance(metrics, dict):
            raise TypeError('`metrics` should be a dict')
        else:
            metrics = dict(metrics)
        metrics.update(kwargs)

        # send the metrics to epoch and step logger
        if self._within_epoch:
            self._epoch_logger.collect(metrics)
        if self._within_step:
            self._step_logger.collect(metrics)

        # now trigger the event of metric collected
        self.on_metrics_collected.fire(metrics)

    def _commit_epoch_stop_time(self):
        if self._epoch_start_time is not None:
            duration = time.time() - self._epoch_start_time
            self.collect(metrics={_EPOCH_TIME_METRIC: duration})
            self._epoch_start_time = None

    def _commit_step_stop_time(self):
        if self._step_start_time is not None:
            duration = time.time() - self._step_start_time
            self.collect(metrics={_STEP_TIME_METRIC: duration})
            self._step_start_time = None

    def make_checkpoint(self):
        """Make a checkpoint."""
        if self._checkpoint_manager is None:
            raise RuntimeError('Checkpoint directory is not configured.')
        self._checkpoint_manager.save()

    def print_stats(self):
        """
        Print the metric statistics as logs.

        Can be called only when an epoch or a step loop has been opened.
        """
        self._require_context()

        if self._within_step:
            self._commit_step_stop_time()
            logger = self._step_logger
        elif self._within_epoch:
            self._commit_epoch_stop_time()
            logger = self._epoch_logger
        else:  # pragma: no cover
            raise RuntimeError('should not touch this branch.')

        self.print(logger.format_logs(clear_stats=False), status_tag=True)

        # fire the metric stats printed event
        metric_stats = {
            key: (c.mean, c.stddev if c.counter > 1 else None)
            for key, c in logger.stats_collectors.items()
            if c.has_value
        }
        self.on_stats_printed.fire(metric_stats)
        logger.clear_stats()

    def _enter(self):
        # restore the checkpoint
        if self._checkpoint_manager is not None:
            checkpoint_dir = None

            if isinstance(self._restore_checkpoint, str):
                checkpoint_dir = self._restore_checkpoint
            elif self._restore_checkpoint:
                checkpoint_dir = \
                    self._checkpoint_manager.latest_checkpoint_dir()

            if checkpoint_dir:
                self._checkpoint_manager.restore(checkpoint_dir)
                self.print(
                    f'Resume training: epoch {self.epoch}, step {self.step}, '
                    f'from checkpoint directory {checkpoint_dir!r}'
                )

        # initialize the eta flags
        if self.epoch == 0 and self.step == 0:
            self._eta.take_snapshot(0.)
        else:
            progress = self.get_progress()
            if progress is not None:
                self._eta.take_snapshot(progress)

        # trigger the event
        self.on_enter_loop.fire()

        # return self as the context object
        return self

    def _exit(self, exc_type, exc_val, exc_tb):
        # trigger the event
        self.on_exit_loop.fire()

        # clear status
        self._steps_per_epoch = None
        self._eta = None

    def _require_context(self):
        self._require_entered()
        if not self._within_epoch and not self._within_step:
            raise RuntimeError('Neither an epoch nor a step loop has been '
                               'entered.')

    @contextmanager
    def timeit(self, metric_name: str):
        """
        Open a context for timing.

        Can be called only when an epoch or a step loop has been opened.

        Args:
            metric_name: Name of the time metric.  Should end with "_time".
        """
        metric_name = str(metric_name).lower()
        if metric_name.rsplit('_', 1)[-1] != 'time':
            raise ValueError(f'`metric_name` does not end with "_time": '
                             f'{metric_name!r}')

        self._require_context()
        start_time = time.time()
        yield
        duration = time.time() - start_time
        self.collect({metric_name: duration})

    @contextmanager
    def collector(self, metric_name: str
                  ) -> Generator[StatisticsCollector, None, None]:
        """
        Get a statistics collector for metric `metric_name`.

        Can be called only when an epoch or a step loop has been opened.

        The mean of the metrics will be added to this train loop after exiting
        the context.  Other statistics will be discarded.

        Args:
            metric_name: The name of the metric.
        """
        self._require_context()
        c = StatisticsCollector()
        yield c
        if c.has_value:
            self.collect(metrics={metric_name: c.mean})

    def iter_epochs(self,
                    limit: Optional[int] = None,
                    count: Optional[int] = None,
                    ) -> Generator[int, None, None]:
        """
        Iterate through the epochs.

        This method can only be called when there's no other epoch loop
        is being iterated.  Furthermore, after exiting this loop, both
        the epoch metrics as well as the step metrics will be cleared.

        If `max_epoch` is configured, it will stop at it.

        Yields:
            The epoch counter (starting from 1).
        """

        def loop_condition():
            return (
                (limit is None or self.epoch < limit) and
                (count is None or counter[0] < count) and
                (self._max_epoch is None or self.epoch < self._max_epoch) and
                (self._max_step is None or self.step < self._max_step)
            )

        self._require_entered()
        if self._within_epoch:
            raise RuntimeError('Another epoch loop has been opened.')

        try:
            counter = [0]

            while loop_condition():
                # note the initial epoch is 0, but our epoch should start from 1
                self.state.epoch += 1
                self._within_epoch = True
                self._epoch_start_time = time.time()

                # execute the epoch
                self.on_enter_epoch.fire(self.epoch)
                yield self.epoch
                self.on_exit_epoch.fire(self.epoch)

                self._commit_epoch_stop_time()
                self._steps_per_epoch = float(self.step) / self.epoch

                # log if epoch matches log_freq
                if self.log_freq.is_epoch_matches(self.epoch):
                    self.print_stats()

                # make checkpoint if epoch matches checkpoint_freq
                if self.checkpoint_freq.is_epoch_matches(self.epoch):
                    self.make_checkpoint()

                counter[0] += 1

        finally:
            self._within_epoch = False
            self._epoch_start_time = None
            self._step_logger.clear_stats()
            self._epoch_logger.clear_stats()

    def iter_steps(self,
                   data_generator: Optional[DataGeneratorType] = None,
                   limit: Optional[int] = None,
                   count: Optional[int] = None,
                   ) -> Generator[Union[int, Tuple[int, Any]], None, None]:
        """
        Iterate through the steps.

        This method can only be called when there's no other step loop
        is being iterated, and an epoch loop is active.

        Args:
            data_generator: Optional iterable data to be yielded at every step.
                This is required if `max_step`, `limit` and `count` are all
                None, so as to prevent an infinite step loop.

        Yields:
            A tuple of ``(step counter, batch data)`` if `data_generator` is
            specified, or otherwise only the step counter.  The step counter
            starts from 1.
        """

        def loop_condition():
            return (
                (limit is None or self.step < limit) and
                (count is None or counter[0] < count) and
                (self._max_step is None or self.step < self._max_step)
            )

        self._require_entered()
        if self._within_step:
            raise RuntimeError('Another step loop has been opened.')
        if data_generator is None and self._max_step is None and \
                limit is None and count is None:
            raise RuntimeError('`data_generator` is required when `max_step`, '
                               '`limit` and `count` are all None, '
                               'so as to prevent an unstoppable loop')

        try:
            if data_generator is not None:
                data_iterator = iter(data_generator)
            else:
                data_iterator = None

            counter = [0]

            while loop_condition():
                # prepare for the step data
                if data_iterator is None:
                    yield_obj = self.step + 1
                    step_data = None
                else:
                    try:
                        step_data = next(data_iterator)
                    except StopIteration:
                        break
                    yield_obj = self.step + 1, step_data

                # prepare internal state for the step
                self.state.step += 1
                self._within_step = True
                self._step_start_time = time.time()

                # execute this step
                self.on_enter_step.fire(self.step, step_data)
                yield yield_obj
                self.on_exit_step.fire(self.step)

                self._commit_step_stop_time()

                # log if step matches log_freq
                if self.log_freq.is_step_matches(self.step):
                    self.print_stats()

                # make checkpoint if step matches checkpoint_freq
                if self.checkpoint_freq.is_step_matches(self.step):
                    self.make_checkpoint()

                counter[0] += 1

        finally:
            self._within_step = False
            self._step_start_time = None

    def run(self,
            train_fn: Callable[..., None],
            data_stream: DataStream):
        """
        Iterate over epochs and steps, executing `train_fn` in each step.

        Args:
            train_fn: The train step function, should receive all arrays
                from `data_stream` as its positional arguments.
            data_stream: The data stream, to generate data for each step.
        """
        return self.run_epochs(train_fn, data_stream)

    def run_epochs(self,
                   train_fn: Callable[..., None],
                   data_stream: DataStream,
                   limit: Optional[int] = None,
                   count: Optional[int] = None):
        """
        Iterate over epochs and steps, executing `train_fn` in each step.

        Unlike :meth:`run()`, this method can control the number of epochs
        to run via `limit` or `count`.  `limit` is the maximum epoch to reach,
        while `count` is the maximum number of epochs to run.
        The difference between these two arguments can be demonstrated by the
        following example:

        >>> def train_fn(x):
        ...    print(x)
        >>> stream = DataStream.int_seq(5, batch_size=3)

        >>> with TrainLoop(max_epoch=3) as loop:  #doctest: +ELLIPSIS
        ...    loop.run_epochs(train_fn, stream, limit=2)
        ...    loop.run_epochs(train_fn, stream, limit=2)
        [0 1 2]
        [3 4]
        [Epoch 1/3, Step 2, ETA ...s] epoch_time: ...s; step_time: ...s (±...s)
        [0 1 2]
        [3 4]
        [Epoch 2/3, Step 4, ETA ...s] epoch_time: ...s; step_time: ...s (±...s)

        >>> with TrainLoop(max_epoch=3) as loop:  # doctest: +ELLIPSIS
        ...    loop.run_epochs(train_fn, stream, count=2)
        ...    loop.run_epochs(train_fn, stream, count=2)
        [0 1 2]
        [3 4]
        [Epoch 1/3, Step 2, ETA ...s] epoch_time: ...s; step_time: ...s (±...s)
        [0 1 2]
        [3 4]
        [Epoch 2/3, Step 4, ETA ...s] epoch_time: ...s; step_time: ...s (±...s)
        [0 1 2]
        [3 4]
        [Epoch 3/3, Step 6, ETA ...s] epoch_time: ...s; step_time: ...s (±...s)

        Args:
            train_fn: The train step function, should receive all arrays
                from `data_stream` as its positional arguments.
            data_stream: The data stream, to generate data for each step.
            limit: Maximum epoch to reach.
            count: Maximum number of epochs to run.
        """
        for _ in self.iter_epochs(limit=limit, count=count):
            for _, batch_data in self.iter_steps(data_stream):
                train_fn(*batch_data)

    def run_steps(self,
                  train_fn: Callable[..., None],
                  data_generator: DataGeneratorType,
                  limit: Optional[int] = None,
                  count: Optional[int] = None):
        """
        Iterate over steps, executing `train_fn` in each step.

        This method can control the number of steps to run via `limit` or
        `count`.  `limit` is the maximum step to reach, while `count` is the
        maximum number of steps to run.
        The difference between these two arguments can be demonstrated by the
        following example:

        >>> def train_fn(x):
        ...    print(x)
        >>> stream = DataStream.int_seq(5, batch_size=3)

        >>> with TrainLoop(max_step=3,  #doctest: +ELLIPSIS
        ...        log_freq='every step') as loop:
        ...    loop.run_steps(train_fn, stream, limit=2)
        ...    loop.run_steps(train_fn, stream, limit=2)
        [0 1 2]
        [Step 1/3, ETA ...s] step_time: ...s
        [3 4]
        [Step 2/3, ETA ...s] step_time: ...s

        >>> with TrainLoop(max_step=3,  #doctest: +ELLIPSIS
        ...        log_freq='every step') as loop:
        ...    loop.run_steps(train_fn, stream, count=2)
        ...    loop.run_steps(train_fn, stream, count=2)
        [0 1 2]
        [Step 1/3, ETA ...s] step_time: ...s
        [3 4]
        [Step 2/3, ETA ...s] step_time: ...s
        [0 1 2]
        [Step 3/3, ETA ...s] step_time: ...s

        Args:
            train_fn: The train step function, should receive all arrays
                from `data_stream` as its positional arguments.
            data_generator: The data stream or generator, to produce the
                data for each step.
            limit: Maximum step to reach.
            count: Maximum number of steps to run.

        Notes:
            Be aware that the step counter will not be reset after each
            epoch!  So if you want to run a number of steps in each epoch,
            you should probably want to use `count` instead of `limit`.
        """
        for _, batch_data in self.iter_steps(
                data_generator, limit=limit, count=count):
            train_fn(*batch_data)

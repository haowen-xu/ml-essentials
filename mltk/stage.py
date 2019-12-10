import time
from enum import Enum
from typing import *

from dataclasses import dataclass

__all__ = [
    'CycleCounter', 'TimedCycleCounter', 'StageType', 'Stage',
    'StageCallbackData', 'StageCallback',
]

from mltk.utils import NOT_SET

MetricsDict = Dict[str, Any]
INDEX_BEFORE_FIRST_CYCLE = -1


class CycleCounter(object):
    """
    Base cycle counter.

    A cycle is one step in a certain type of loop.  For example, in a train
    loop, there are typically multiple epoch cycles; and in each epoch,
    there are multiple batch cycles.

    When cycles are nested, one step of the external cycle is called
    a whole loop of the internal cycle.
    """

    __slots__ = ('index', 'total', 'avg_total', 'is_active', 'loop_index')

    def __init__(self,
                 index: int = INDEX_BEFORE_FIRST_CYCLE,
                 total: Optional[int] = None,
                 loop_index: int = 0):
        # index of the current cycle, start from 0.
        self.index: int = index
        # total number of cycles to run in each loop.
        self.total: Optional[int] = total
        # index of the current loop, start from 0.
        self.loop_index: int = loop_index
        # average total number of cycles in each loop.
        self.avg_total: Optional[float] = None
        # is this cycle entered but not exited now?
        self.is_active: bool = False

    def enter(self, index: Optional[int] = None):
        """
        Enter one cycle.

        Args:
            index: The index of this cycle.  If not specified, will
                increase ``self.index`` by 1.
        """
        if index is not None:
            self.index = index
        else:
            self.index += 1
        self.is_active = True

    def exit(self):
        """Exit this cycle."""
        self.is_active = False

    def next_loop(self):
        """
        Enter the next loop.

        This will update the `avg_total` estimation by the current `index`,
        and then set cycle `index` to zero.
        """
        # reset index counter
        this_total = self.index + 1
        self.index = INDEX_BEFORE_FIRST_CYCLE

        # increase loop index
        self.loop_index += 1

        # update the `avg_total` counter
        n = self.loop_index
        if self.avg_total is None:
            self.avg_total = this_total
        elif abs(self.avg_total - this_total) > 1e-7:
            self.avg_total = ((n - 1.) / n * self.avg_total +
                              float(this_total) / n)

    def estimated_cycles_ahead(self,
                               count_this_cycle: bool = NOT_SET
                               ) -> Optional[float]:
        """
        Get the estimated cycles ahead in the current loop.

        This method will use `avg_total` prior than `total`.  If `is_open`
        is :obj:`True`, then the current cycle will also be counted.

        Args:
            count_this_cycle: Whether or not to count the current cycle?
                If :obj:`True`, the current cycle (i.e., ``self.index``)
                will also be counted as cycles ahead.  Otherwise the current
                cycle will not be counted.  If not specified, will count
                this cycle only if ``self.is_active == True``.

        Returns:
            The estimated cycles ahead, may be a float number.
            Will be :obj:`None` if the total cycles is not known.
        """
        if count_this_cycle is NOT_SET:
            count_this_cycle = self.is_active
        total = self.avg_total if self.avg_total is not None else self.total
        if total is not None:
            ahead = total - self.index
            if not count_this_cycle:
                ahead -= 1
            return ahead

    def __str__(self):
        s = f'/{self.total}' if self.total is not None else ''
        return f'{self.index + 1}{s}'


class TimedCycleCounter(CycleCounter):
    """A cycle counter with timer."""

    __slots__ = (
        'index', 'total', 'avg_total', 'is_active', 'loop_index',
        'start_timestamp', 'end_timestamp', 'last_cycle_time', 'avg_cycle_time',
        '_avg_cycle_time_n_estimates'
    )

    def __init__(self,
                 index: int = INDEX_BEFORE_FIRST_CYCLE,
                 total: Optional[int] = None,
                 loop_index: int = 0):
        super().__init__(index=index, total=total, loop_index=loop_index)
        self.start_timestamp: Optional[float] = None
        self.end_timestamp: Optional[float] = None
        self.avg_cycle_time: Optional[float] = None
        self.last_cycle_time: Optional[float] = None  # execution time of the last finished cycle
        self._avg_cycle_time_n_estimates: int = 0

    def enter(self, index: Optional[int] = None):
        super().enter(index)
        self.start_timestamp = time.time()

    def pre_exit(self):
        self.end_timestamp = time.time()
        self.last_cycle_time = self.end_timestamp - self.start_timestamp

        # update the `avg_cycle_time` estimation
        self._avg_cycle_time_n_estimates += 1
        if self.avg_cycle_time is None:
            self.avg_cycle_time = self.last_cycle_time
        else:
            n = self._avg_cycle_time_n_estimates
            self.avg_cycle_time = (
                (n - 1.) / n * self.avg_cycle_time +
                float(self.last_cycle_time) / n
            )

    def estimated_time_ahead(self,
                             count_this_cycle: bool = NOT_SET
                             ) -> Optional[float]:
        """
        Get the estimated time ahead (ETA).

        Args:
            count_this_cycle: Whether or not to count the current cycle?
                If :obj:`True`, the current cycle (i.e., ``self.index``)
                will also be counted as cycles ahead.  Otherwise the current
                cycle will not be counted.  If not specified, will count
                this cycle only if ``self.is_active == True``.

        Returns:
            The estimated time ahead, in seconds.
            Will be :obj:`None` if the total cycles is not known.
        """
        if self.avg_cycle_time is not None:
            cycles_ahead = self.estimated_cycles_ahead(count_this_cycle)
            if cycles_ahead is not None:
                return cycles_ahead * self.avg_cycle_time


class StageType(str, Enum):
    """Machine learning experiment stage types."""

    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'
    PREDICT = 'predict'


class Stage(object):
    """
    Base class of a machine learning stage.

    A :class:`Stage` represents a certain (large) step of a machine learning
    experiment, which uses a given dataset for one or more epochs.
    The :class:`Stage` class maintains the epoch and batch counters, and
    organizes callbacks for the stage.
    """

    type: StageType
    epoch: Optional[TimedCycleCounter]
    batch: TimedCycleCounter
    batch_size: Optional[int]  # batch size, i.e., maximum number of samples in each batch
    data_count: Optional[int]  # number of data samples
    global_step: Optional[int]  # the global step counter
    callbacks: List['StageCallback']
    known_metrics: Tuple[str, ...]

    _current_batch_size: Optional[int] = None  # the size of the current active batch
    _current_epoch_size: Optional[int] = None  # the size of the current active epoch

    running: bool = False
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    stage_time: Optional[float] = None

    def __init__(self,
                 type: StageType,
                 initial_epoch: int = INDEX_BEFORE_FIRST_CYCLE,
                 total_epochs: Optional[int] = None,
                 initial_batch: int = INDEX_BEFORE_FIRST_CYCLE,
                 total_batches: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 data_count: Optional[int] = None,
                 global_step: Optional[int] = None,
                 callbacks: Optional[Sequence['StageCallback']] = None,
                 known_metrics: Optional[Sequence[str]] = None):

        epoch = TimedCycleCounter(initial_epoch, total=total_epochs) \
            if type == StageType.TRAIN else None
        batch = TimedCycleCounter(initial_batch, total=total_batches)
        callbacks = list(callbacks or ())

        self.type = type
        self.epoch = epoch
        self.batch = batch
        self.batch_size = batch_size
        self.data_count = data_count
        self.global_step = global_step
        self.callbacks = callbacks

        if known_metrics is not None:
            known_metrics = tuple([self.add_metric_prefix(k)
                                   for k in known_metrics])
        self.known_metrics = known_metrics

    @property
    def name(self) -> str:
        """
        Get the name of the stage.

        Returns:
            One of: {"train", "validation", "test", "predict"}.
        """
        return self.type.value

    __METRIC_PREFIXES = {
        StageType.TRAIN: ('train_', ''),
        StageType.VALIDATION: ('val_', 'valid_'),
        StageType.TEST: ('test_',),
        StageType.PREDICT: ('predict_',)
    }

    @property
    def metric_prefixes(self) -> Tuple[str, ...]:
        """
        Get the allowed prefixes of the metric names for this stage.

        The correspondence between the stage type and the allowed prefixes
        is listed as follows:

        *  StageType.TRAIN: "train_", ""
        *  StageType.VALIDATION: "val_", "valid_"
        *  StageType.TEST: "test_"
        *  StageType.PREDICT: "predict_"
        """
        return self.__METRIC_PREFIXES[self.type]

    @property
    def metric_prefix(self) -> str:
        """Get the preferred metric name prefix."""
        return self.metric_prefixes[0]

    def add_metric_prefix(self, name: str) -> str:
        """
        Add stage metric prefix to the metric name, if absent.

        Args:
            name: The original metric name.

        Returns:
            The processed metric name.
        """
        if not any(name.startswith(pfx) for pfx in self.metric_prefixes):
            name = f'{self.metric_prefixes[0]}{name}'
        return name

    @property
    def total_epochs(self) -> Optional[int]:
        return self.epoch and self.epoch.total

    @property
    def total_batches(self) -> Optional[int]:
        return self.batch and self.batch.total

    def get_eta(self) -> Optional[float]:
        if self.epoch is not None:
            # get the total batches
            total_batches = self.batch.avg_total or self.batch.total

            # estimate the epoch time
            epoch_time = self.epoch.avg_cycle_time
            if epoch_time is None:
                batch_time = self.batch.avg_cycle_time
                if batch_time is None or total_batches is None:
                    return None  # no way to estimate epoch time, return None
                epoch_time = batch_time * total_batches

            # estimate the epoch ahead
            epoch_ahead = self.epoch.estimated_cycles_ahead()
            if epoch_ahead is None:
                return None

            if total_batches:
                batch_ahead = self.batch.estimated_cycles_ahead()
                if batch_ahead is not None:
                    epoch_ahead = epoch_ahead - 1 + \
                        float(batch_ahead) / total_batches

            # now compute eta
            return epoch_ahead * epoch_time

        else:
            return self.batch.estimated_time_ahead()

    def get_progress_str(self) -> str:
        if self.epoch is not None:
            return f'Epoch {self.epoch}, Batch {self.batch}'
        else:
            return f'Batch {self.batch}'

    def enter(self):
        if self.start_timestamp is not None:
            raise RuntimeError('`BaseStage` is neither re-entrant, nor '
                               'reusable.')

        # initialize the statuses
        self.running = True
        self.start_timestamp = time.time()
        self.end_timestamp = None

        # call the callbacks
        event_name = f'on_{self.name}_begin'
        event_data = StageCallbackData(
            stage=self,
            index=None,
            size=None,
            exc_time=None,
            metrics=None
        )
        for cb in self.callbacks:
            cb.on_stage_begin(event_data)
            getattr(cb, event_name)(event_data)

    def exit(self, metrics: Optional[MetricsDict] = None):
        try:
            self.end_timestamp = time.time()
            exc_time = self.end_timestamp - self.start_timestamp

            # call the callbacks
            event_name = f'on_{self.name}_end'
            event_data = StageCallbackData(
                stage=self,
                index=None,
                size=None,
                exc_time=exc_time,
                metrics=metrics
            )
            for cb in self.callbacks:
                cb.on_stage_end(event_data)
                getattr(cb, event_name)(event_data)
        finally:
            self.running = False

    def enter_epoch(self,
                    epoch: Optional[int] = None,
                    epoch_size: Optional[int] = None):
        if self.epoch is None:
            raise RuntimeError(f'Stage {self!r} does not have an epoch '
                               f'counter.')
        self.epoch.enter(epoch)
        self._current_epoch_size = epoch_size

        # call the callbacks
        event_name = f'on_{self.name}_epoch_begin'
        event_data = StageCallbackData(
            stage=self,
            index=self.epoch.index,
            size=self._current_epoch_size,
            exc_time=None,
            metrics=None
        )
        for cb in self.callbacks:
            cb.on_epoch_begin(event_data)
            getattr(cb, event_name)(event_data)

    def exit_epoch(self,
                   metrics: Optional[MetricsDict] = None):
        if self.epoch is None:
            raise RuntimeError(f'Stage {self!r} does not have an epoch '
                               f'counter.')
        self.epoch.pre_exit()

        try:
            # call the callbacks
            event_name = f'on_{self.name}_epoch_end'
            event_data = StageCallbackData(
                stage=self,
                index=self.epoch.index,
                size=self._current_epoch_size,
                exc_time=self.epoch.last_cycle_time,
                metrics=metrics
            )
            for cb in self.callbacks:
                cb.on_epoch_end(event_data)
                getattr(cb, event_name)(event_data)

        finally:
            self.epoch.exit()
            self.batch.next_loop()
            self._current_epoch_size = None

    def enter_batch(self,
                    batch: Optional[int] = None,
                    batch_size: Optional[int] = None):
        self.batch.enter(batch)
        if self.global_step is not None:
            self.global_step += 1
        self._current_batch_size = batch_size

        # call the callbacks
        event_name = f'on_{self.name}_batch_begin'
        event_data = StageCallbackData(
            stage=self,
            index=self.batch.index,
            size=self._current_batch_size,
            exc_time=None,
            metrics=None
        )
        for cb in self.callbacks:
            cb.on_batch_begin(event_data)
            getattr(cb, event_name)(event_data)

    def exit_batch(self,
                   metrics: Optional[MetricsDict] = None):
        self.batch.pre_exit()
        try:
            # call the callbacks
            event_name = f'on_{self.name}_batch_end'
            event_data = StageCallbackData(
                stage=self,
                index=self.batch.index,
                size=self._current_batch_size,
                exc_time=self.batch.last_cycle_time,
                metrics=metrics
            )
            for cb in self.callbacks:
                cb.on_batch_end(event_data)
                getattr(cb, event_name)(event_data)

        finally:
            self.batch.exit()
            self._current_batch_size = None


@dataclass
class StageCallbackData(object):
    """
    Data carried by a cycle begin/end event from :class:`StageCallback`.
    """

    __slots__ = ('stage', 'index', 'size', 'exc_time', 'metrics')

    stage: Stage
    """The stage that calls the callback."""

    index: Optional[int]
    """Index of the epoch or batch."""

    size: Optional[int]
    """The size of the batch."""

    exc_time: Optional[float]
    """Execution time of the stage/epoch/batch, available at the cycle end."""

    metrics: Optional[MetricsDict]
    """Metrics dict, available at the cycle end."""


class StageCallback(object):
    """Base class of a callback for a machine learning stage."""

    ##################
    # general events #
    ##################
    def on_stage_begin(self, data: StageCallbackData):
        pass

    def on_stage_end(self, data: StageCallbackData):
        pass

    def on_epoch_begin(self, data: StageCallbackData):
        pass

    def on_epoch_end(self, data: StageCallbackData):
        pass

    def on_batch_begin(self, data: StageCallbackData):
        pass

    def on_batch_end(self, data: StageCallbackData):
        pass

    ################
    # train events #
    ################
    def on_train_begin(self, data: StageCallbackData):
        pass

    def on_train_end(self, data: StageCallbackData):
        pass

    def on_train_epoch_begin(self, data: StageCallbackData):
        pass

    def on_train_epoch_end(self, data: StageCallbackData):
        pass

    def on_train_batch_begin(self, data: StageCallbackData):
        pass

    def on_train_batch_end(self, data: StageCallbackData):
        pass

    #####################
    # validation events #
    #####################
    def on_validation_begin(self, data: StageCallbackData):
        pass

    def on_validation_end(self, data: StageCallbackData):
        pass

    def on_validation_batch_begin(self, data: StageCallbackData):
        pass

    def on_validation_batch_end(self, data: StageCallbackData):
        pass

    ###############
    # test events #
    ###############
    def on_test_begin(self, data: StageCallbackData):
        pass

    def on_test_end(self, data: StageCallbackData):
        pass

    def on_test_batch_begin(self, data: StageCallbackData):
        pass

    def on_test_batch_end(self, data: StageCallbackData):
        pass

    ##################
    # predict events #
    ##################
    def on_predict_begin(self, data: StageCallbackData):
        pass

    def on_predict_end(self, data: StageCallbackData):
        pass

    def on_predict_batch_begin(self, data: StageCallbackData):
        pass

    def on_predict_batch_end(self, data: StageCallbackData):
        pass

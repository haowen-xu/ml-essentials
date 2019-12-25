from dataclasses import dataclass
from itertools import chain
from typing import *

import numpy as np

from .batch_agg import *
from .callbacks import *
from .data import DataStream
from .events import Event, EventHost
from .mlstorage import ExperimentDoc
from .stage import Stage, StageType
from .utils import to_number_or_numpy, get_array_shape, ALL, NOT_SET, DocInherit

__all__ = [
    'TrainLoop', 'ValidationLoop', 'TestLoop', 'PredictLoop',
]

BatchArrayTuple = Tuple[np.ndarray, ...]
BatchArrays = Union[BatchArrayTuple, List[np.ndarray]]
BatchGeneratorType = Generator[
    Union[int, Tuple[int, BatchArrayTuple]],
    None,
    None
]
AggregatorDict = Mapping[str, BatchAggregator]


class _BaseLoopEventCallback(Callback):

    # a sufficiently small priority, such that it should run before almost
    # all callbacks
    priority = -999999

    loop: 'BaseLoop'
    stage: Stage

    def __init__(self,
                 loop: 'BaseLoop',
                 stage: Stage):
        self.loop = loop
        self.stage = stage

    def on_stage_begin(self, data: CallbackData):
        if data.stage == self.stage:
            self.loop.on_begin.fire()

    def on_stage_end(self, data: CallbackData):
        if data.stage == self.stage:
            self.loop.on_end.fire()

    def on_batch_begin(self, data: CallbackData):
        if data.stage == self.stage:
            self.loop.on_batch_begin.fire()

    def on_batch_end(self, data: CallbackData):
        if data.stage == self.stage:
            self.loop.on_batch_end.fire()


class _TrainLoopEventCallback(_BaseLoopEventCallback):

    loop: 'TrainLoop'

    def on_epoch_begin(self, data: CallbackData):
        if data.stage == self.stage:
            self.loop.on_epoch_begin.fire()

    def on_epoch_end(self, data: CallbackData):
        if data.stage == self.stage:
            self.loop.on_epoch_end.fire()


class BaseLoop(metaclass=DocInherit):

    _LoopEventCallbackClass = _BaseLoopEventCallback
    """Callback class that fires the loop events."""

    RUN_BATCHES_DEFAULT_METRICS = ALL
    """Default value for the `metrics` arg in :meth:`run_batches()`."""

    RUN_BATCHES_DEFAULT_OUTPUTS = ()
    """Default value for the `outputs` arg in :meth:`run_batches()`."""

    _callbacks: CallbackList

    logger: LoggerCallback
    """
    The last :class:`LoggerCallback` granted to this loop.

    If no :class:`LoggerCallback` is granted, then this should be the
    auto-created logger callback instance.
    """

    _stage: Stage  # the active stage
    _remote_doc: Optional[ExperimentDoc]
    _batch_metrics: Dict[str, Any]
    _epoch_metrics: Dict[str, Any]
    _stage_metrics: Dict[str, Any]

    events: EventHost
    """The event host of this loop object."""

    on_begin: Event
    """Event triggered when loop begins, with signature ``() -> None``."""

    on_end: Event
    """Event triggered when loop ends, with signature ``() -> None``."""

    on_batch_begin: Event
    """Event triggered when batch begins, with signature ``() -> None``."""

    on_batch_end: Event
    """Event triggered when batch ends, with signature ``() -> None``."""

    def __init__(self,
                 stage: Stage,
                 remote_doc: Optional[ExperimentDoc] = None,
                 callbacks: Sequence[Callback] = ()):
        """
        Construct a new :class:`BaseLoop`.

        Args:
            stage: The stage object.  Note that the `callbacks` list of
                `stage` will be altered.  Do not share stages between
                different loops.
            remote_doc: The experiment document object.
            callbacks: The callbacks.
        """
        # merge `callbacks` with `stage.callbacks`, sort them into proper
        # order, and add default callbacks if not given.
        self._callbacks = CallbackList.new(
            list(chain(stage.callbacks, callbacks)),
            logger_factory=lambda: LoggerCallback(remote_doc=remote_doc)
        )
        self.logger = next(cb for cb in reversed(self._callbacks)
                           if isinstance(cb, LoggerCallback))
        stage.callbacks = CallbackList(
            [self._LoopEventCallbackClass(self, stage)] + self._callbacks)

        self._stage = stage
        self._remote_doc = remote_doc
        self._batch_metrics = {}
        self._epoch_metrics = {}
        self._stage_metrics = {}

        # bind the events of this object
        self.events = EventHost()
        self.on_begin = self.events['on_begin']
        self.on_end = self.events['on_end']
        self.on_batch_begin = self.events['on_batch_begin']
        self.on_batch_end = self.events['on_batch_end']

    @property
    def batch(self) -> int:
        return self._stage.batch.index

    @property
    def max_batch(self) -> Optional[int]:
        return self._stage.batch.total

    def add_metrics(self,
                    metrics_: Optional[Dict[str, Any]] = None,
                    **kwargs: Any) -> None:
        """
        Add metrics to the loop.

        Note this method will not affect the metrics and outputs aggregated
        by :class:`BatchOutputAggregator` in :meth:`run_batches()`.

        Args:
            metrics_, \\**kwargs: The metrics to be collected.
        """
        def collect(target: Dict[str, Any]):
            if metrics_:
                for key, val in metrics_.items():
                    target[key] = to_number_or_numpy(val)
            if kwargs:
                for key, val in kwargs.items():
                    target[key] = to_number_or_numpy(val)

        if self._stage.batch.is_active:
            collect(self._batch_metrics)
        elif self._stage.epoch is not None and self._stage.epoch.is_active:
            collect(self._epoch_metrics)
        else:
            collect(self._stage_metrics)

    def _iter_batches(self,
                      data_generator: Optional[Iterable[BatchArrays]] = None,
                      limit: Optional[int] = None,
                      count: Optional[int] = None,
                      ) -> BatchGeneratorType:
        # inspect the data generator to complete the total number of batches,
        # if `limit` and `count` is not specified
        if data_generator is not None and count is None and limit is None:
            g_info = inspect_data_generator(data_generator)
            if g_info.batch_count is not None and \
                    self._stage.batch.total is None:
                self._stage.batch.total = (self._stage.batch.index +
                                           g_info.batch_count)

        # get the upper limit of `batch.index`
        if limit is not None:
            batch_limit = limit
        elif count is not None:
            # `+1` because `batch.index` points to the previously completed
            # batch.
            batch_limit = self._stage.batch.index + count
        else:
            batch_limit = self._stage.batch.total

        if self._stage.batch.total is not None:
            batch_limit = min(self._stage.batch.total, batch_limit)

        # convert `data_generator` into iterator
        close_data_iterator = False
        if data_generator is not None:
            data_iterator = iter(data_generator)
            if isinstance(data_generator, DataStream):
                # we've just obtained a temporary iterator from the DataStream,
                # thus it's our responsibility to close it.
                close_data_iterator = True
        else:
            data_iterator = None

        # now run the loop
        try:
            if data_iterator is not None:
                while not self._stage.termination_requested and \
                        (batch_limit is None or
                         self._stage.batch.index < batch_limit):
                    try:
                        batch_data = next(data_iterator)
                    except StopIteration:
                        break

                    # check batch data and inspect batch size
                    if not isinstance(batch_data, (tuple, list)) or \
                            not batch_data:
                        raise ValueError(
                            f'`data_generator` did not yield a non-empty tuple '
                            f'or list of arrays: got {batch_data!r}'
                        )
                    batch_size = len(batch_data[0])

                    # now run the batch
                    self._batch_metrics.clear()
                    self._stage.enter_batch(batch_size=batch_size)
                    try:
                        yield self._stage.batch.index, batch_data
                    finally:
                        self._stage.exit_batch(self._batch_metrics)

            else:
                while self._stage.batch.index < batch_limit:
                    self._batch_metrics.clear()
                    self._stage.enter_batch()
                    try:
                        yield self._stage.batch.index
                    finally:
                        self._stage.exit_batch(self._batch_metrics)
        finally:
            if close_data_iterator:
                data_iterator.close()

    def iter_batches(self,
                     data_generator: Optional[Iterable[BatchArrays]] = None,
                     limit: Optional[int] = None,
                     count: Optional[int] = None,
                     ) -> BatchGeneratorType:
        """
        Iterate through the batches.

        Args:
            data_generator: Mini-batch data generator, yielding tuple of arrays.
            limit: The maximum batch index to reach, i.e., ``index <= limit``
                is a loop constraint on the batch counter.
            count: The maximum number of batches to run.

        Yields:
            (int, Tuple[np.ndarray, ...]): The batch index and mini-batch
                arrays, if `data_generator` is specified.
            int: The batch index, if `data_generator` is not specified.

        """
        # check the context
        if not self._stage.is_active:
            raise RuntimeError('The loop context must be entered before '
                               'calling `iter_batches()`.')
        if self._stage.batch.is_active:
            raise RuntimeError('`iter_batches()` cannot be called when a '
                               'batch is currently running.')

        # check the arguments
        if count is not None and limit is not None:
            raise ValueError('`count` and `limit` cannot be both specified.')

        # we do not allow infinite loop
        if data_generator is None and count is None and limit is None and \
                self._stage.batch.total is None:
            raise ValueError(
                'Any one of `data_generator`, `limit` or `count` is required '
                'to be specified when `max_batch` is not configured for '
                'the loop.')

        return self._iter_batches(
            data_generator=data_generator,
            limit=limit,
            count=count,
        )

    def _complete_metrics_and_outputs_arg(self, metrics, outputs):
        if metrics is NOT_SET and outputs == ALL:
            metrics = ()
        elif outputs is NOT_SET and metrics == ALL:
            outputs = ()
        else:
            if metrics is NOT_SET:
                metrics = self.RUN_BATCHES_DEFAULT_METRICS
            if outputs is NOT_SET:
                outputs = self.RUN_BATCHES_DEFAULT_OUTPUTS
        return metrics, outputs

    def run_batches(self,
                    fn: Callable[..., Optional[Dict[str, Any]]],
                    data_generator: Iterable[BatchArrays],
                    limit: Optional[int] = None,
                    count: Optional[int] = None,
                    metrics: Union[Sequence[str], type(ALL)] = NOT_SET,
                    outputs: Union[Sequence[str], type(ALL)] = NOT_SET,
                    aggregators: Optional[Mapping[str, BatchAggregator]] = None,
                    excludes: Sequence[str] = ()
                    ) -> Optional[Dict[str, Any]]:
        """
        Run batches with the specified batch function `fn`.

        Args:
            fn: The batch function to execute at each batch.
                The signature of `fn` should be ``(*arrays) -> None` or
                ``(*arrays) -> Dict[str, Any]``, which consumes the batch
                arrays produced by `data_generator`, and (maybe) returns the
                batch metrics and outputs.
            data_generator: Mini-batch data generator, yielding tuple of arrays.
            limit: The maximum batch index to reach, i.e., ``index <= limit``
                is a loop constraint on the batch counter.
            count: The maximum number of batches to run.
            metrics: Names of metrics produced by `fn`.  These metrics will
                be aggregated by ``BatchAggregator('AVERAGE', axis=None)``,
                reported by ``self.logger``, and returned by this method.
                Defaults to ``SELF.RUN_BATCHES_DEFAULT_METRICS``.
            outputs: Names of outputs produced by `fn`.  These outputs will
                be aggregated by ``BatchAggregator('CONCAT', axis=0)``,
                and returned by this method.
                Defaults to ``SELF.RUN_BATCHES_DEFAULT_OUTPUTS``.
            aggregators: Dict from name to custom batch aggregators.
            excludes: The names to exclude, of items produced by `fn`.
                If a name is excluded, it will not be collected by any
                :class:`BatchAggregator`.

        Returns:
            The aggregated metrics and outputs.
        """
        metrics, outputs = \
            self._complete_metrics_and_outputs_arg(metrics, outputs)

        # the BatchAggregatorDict
        agg_dict = BatchAggregatorDict.new(
            metrics=metrics,
            outputs=outputs,
            aggregators=aggregators,
            excludes=excludes,
            stage_type=self._stage.type,
        )

        # the function to add metric prefix
        add_metric_prefix = self._stage.type.add_metric_prefix

        # now run the batches
        g = self.iter_batches(data_generator, limit=limit, count=count)
        try:
            for batch, batch_data in g:
                batch_size = get_array_shape(batch_data[0])[0]
                fn_out = fn(*batch_data)
                if fn_out is not None:
                    if not isinstance(fn_out, dict):
                        raise TypeError(f'The output of `fn` is expected to be '
                                        f'a dict, but got {fn_out!r}')

                    fn_out = {
                        add_metric_prefix(k): to_number_or_numpy(v)
                        for k, v in fn_out.items()
                    }
                    metrics = {}
                    for key, val in fn_out.items():
                        agg = agg_dict.get(key)
                        if agg is not None:
                            size = batch_size if np.shape(val) == () else 1.
                            agg.add(val, weight=size)
                            # For metrics collected by
                            # ``BatchAggregator('AVERAGE', None)``, we also add
                            # them to the batch metrics.
                            if agg.mode == BatchAggregationMode.AVERAGE and \
                                    agg.axis is None:
                                metrics[key] = np.mean(val)
                    self.add_metrics(metrics)
        finally:
            g.close()

        # return the aggregated results
        if len(agg_dict) > 0:
            return {k: v.get() for k, v in agg_dict.items()}

    def __enter__(self):
        if self._stage.is_active:
            raise RuntimeError(f'{self.__class__.__qualname__} is not '
                               f're-entrant.')
        self._stage_metrics.clear()
        self._stage.enter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stage.exit(self._stage_metrics)


class TrainLoop(BaseLoop):

    _LoopEventCallbackClass = _TrainLoopEventCallback

    only_batch: bool
    """Whether or not this train loop only runs batches, without epochs?"""

    on_epoch_begin: Event
    """Event triggered when epoch begins, with signature ``() -> None``."""

    on_epoch_end: Event
    """Event triggered when epoch ends, with signature ``() -> None``."""

    def __init__(self,
                 max_epoch: Optional[int] = None,
                 max_batch: Optional[int] = None,
                 only_batch: bool = False,
                 remote_doc: Optional[ExperimentDoc] = None,
                 callbacks: Sequence[Callback] = ()):
        """
        Construct a new :class:`TrainLoop`.

        Args:
            max_epoch: The maximum index for the epoch counter to reach.
            max_batch: The maximnum index for the batch counter to reach.
            only_batch: Whether or not to iterate only through
                batches, without explicitly iterating through epochs.
                If :obj:`True`, will open an epoch automatically when entering
                the loop, and closing the epoch when exiting the loop.
            remote_doc: The experiment document object.
            callbacks: The callbacks.
        """
        only_batch = bool(only_batch)
        if only_batch and max_epoch is not None:
            raise ValueError('`epochs` must not be specified when '
                             '`only_batch` is set to True.')

        super().__init__(
            stage=Stage(
                type=StageType.TRAIN,
                max_epoch=max_epoch,
                max_batch=max_batch,
            ),
            remote_doc=remote_doc,
            callbacks=callbacks,
        )
        self.only_batch = only_batch
        self.on_epoch_begin = self.events['on_epoch_begin']
        self.on_epoch_end = self.events['on_epoch_end']

    @property
    def epoch(self):
        return self._stage.epoch.index

    @property
    def max_epoch(self):
        return self._stage.epoch.total

    def iter_batches(self,
                     data_generator: Optional[Iterable[BatchArrays]] = None,
                     limit: Optional[int] = None,
                     count: Optional[int] = None
                     ) -> BatchGeneratorType:
        if not self._stage.epoch.is_active:
            raise RuntimeError(
                'The batch loop can only be open inside an epoch loop.  '
                'Did you forget to call `iter_epochs()`?'
            )
        return super().iter_batches(
            data_generator=data_generator,
            limit=limit,
            count=count,
        )

    def _iter_epochs(self,
                     limit: Optional[int] = None,
                     count: Optional[int] = None
                     ) -> Generator[int, None, None]:
        # get the upper limit of `batch.index`
        if limit is not None:
            epoch_limit = limit
        elif count is not None:
            # see `iter_batches()` for the reason of `+1`
            epoch_limit = self._stage.epoch.index + count
        else:
            epoch_limit = self._stage.epoch.total

        if self._stage.epoch.total is not None:
            epoch_limit = min(self._stage.epoch.total, epoch_limit)

        # now run the loop
        while not self._stage.termination_requested and \
                self._stage.epoch.index < epoch_limit:
            self._epoch_metrics.clear()
            self._stage.enter_epoch()
            try:
                yield self._stage.epoch.index
            finally:
                self._stage.exit_epoch(self._epoch_metrics)

    def iter_epochs(self,
                    limit: Optional[int] = None,
                    count: Optional[int] = None
                    ) -> Generator[int, None, None]:
        """
        Iterate through the batches.

        Args:
            limit: The maximum epoch index to reach, i.e., ``index <= limit``
                is a loop constraint on the epoch counter.
            count: The maximum number of epochs to run.

        Yields:
            int: The epoch index.
        """
        # check the context
        if self.only_batch:
            raise RuntimeError('The loop is configured with `only_batch = True`'
                               ', thus `iter_epochs()` is prohibited.')
        if not self._stage.is_active:
            raise RuntimeError('The loop context must be entered before '
                               'calling `iter_epochs()`.')
        if self._stage.epoch.is_active:
            raise RuntimeError('`iter_epochs()` is not re-entrant.')

        # check the arguments
        if count is not None and limit is not None:
            raise ValueError('`count` and `limit` cannot be both specified.')

        # we do not allow infinite loop
        if limit is None and count is None and self._stage.epoch.total is None:
            raise ValueError(
                'Either `limit` or `count` is required to be specified when '
                '`max_epoch` is not configured for the loop.')

        return self._iter_epochs(limit=limit, count=count)

    def run_epochs(self,
                   fn: Callable[..., Optional[Dict[str, Any]]],
                   data_generator: Iterable[BatchArrays],
                   limit: Optional[int] = None,
                   count: Optional[int] = None,
                   metrics: Union[Sequence[str], type(ALL)] = NOT_SET,
                   excludes: Sequence[str] = ()
                   ) -> None:
        """
        Run epochs and the batches in each epoch with the specified batch
        function `fn`.

        Args:
            fn: The batch function to execute at each batch.
                The signature of `fn` should be ``(*arrays) -> None` or
                ``(*arrays) -> Dict[str, Any]``, which consumes the batch
                arrays produced by `data_generator`, and (maybe) returns the
                batch metrics.
            data_generator: Mini-batch data generator, yielding tuple of arrays.
            limit: The maximum epoch index to reach, i.e., ``index <= limit``
                is a loop constraint on the epoch counter.
            count: The maximum number of epochs to run.
            metrics: Names of metrics produced by `fn`.  These metrics will
                be aggregated by ``BatchAggregator('AVERAGE', axis=None)``,
                and reported by ``self.logger``.
            excludes: The names to exclude, of items produced by `fn`.
                If a name is excluded, it will not be collected by any
                :class:`BatchAggregator`.

        Notes:
            Unlike :meth:`run_batches()`, this method will not return the
            collected metrics.  Consider to use :meth:`run_batches()` with
            explicit epoch loop if you need to obtain the metrics.
        """
        g = self.iter_epochs(limit=limit, count=count)
        try:
            for _ in g:
                self.run_batches(
                    fn, data_generator, metrics=metrics, excludes=excludes)
        finally:
            g.close()

    def run(self,
            fn: Callable[..., Optional[Dict[str, Any]]],
            data_generator: Iterable[BatchArrays],
            metrics: Union[Sequence[str], type(ALL)] = NOT_SET,
            excludes: Sequence[str] = ()
            ) -> Optional[Dict[str, Any]]:
        """
        Run the train loop.

        Args:
            fn: The batch function to execute at each batch.
                The signature of `fn` should be ``(*arrays) -> None` or
                ``(*arrays) -> Dict[str, Any]``, which consumes the batch
                arrays produced by `data_generator`, and (maybe) returns the
                batch metrics.
            data_generator: Mini-batch data generator, yielding tuple of arrays.
            metrics: Names of metrics produced by `fn`.  These metrics will
                be aggregated by ``BatchAggregator('AVERAGE', axis=None)``,
                and reported by ``self.logger``.
            excludes: The names to exclude, of items produced by `fn`.
                If a name is excluded, it will not be collected by any
                :class:`BatchAggregator`.

        Returns:
            If ``self.only_batch == True``, then the collected metrics will
            be returned.  Otherwise the return value will alwasy be :obj:`None`.
        """
        run_fn = self.run_batches if self.only_batch else self.run_epochs
        if not self._stage.is_active:
            with self:
                return run_fn(
                    fn, data_generator, metrics=metrics, excludes=excludes)
        else:
            return run_fn(
                fn, data_generator, metrics=metrics, excludes=excludes)

    def validation(self) -> 'ValidationLoop':
        """
        Construct a new :class:`ValidationLoop` that inherits callbacks and
        other states from this train loop.

        This is the recommended way to obtain a validation loop inside a train loop.
        """
        return ValidationLoop(
            remote_doc=self._remote_doc,
            callbacks=self._callbacks
        )

    def test(self) -> 'TestLoop':
        """
        Construct a new :class:`TestLoop` that inherits callbacks and
        other states from this train loop.

        This is the recommended way to obtain a test loop inside a train loop.
        """
        return TestLoop(
            remote_doc=self._remote_doc,
            callbacks=self._callbacks
        )

    def predict(self) -> 'PredictLoop':
        """
        Construct a new :class:`PredictLoop` that inherits callbacks and
        other states from this train loop.

        This is the recommended way to obtain a predict loop inside a train loop.
        """
        return PredictLoop(
            remote_doc=self._remote_doc,
            callbacks=self._callbacks
        )

    def __enter__(self):
        super().__enter__()
        # open the first epoch if `only_batches` is True
        if self.only_batch:
            self._stage.enter_epoch(1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.only_batch:
            self._stage.exit_epoch(self._epoch_metrics)
        return super().__exit__(exc_type, exc_val, exc_tb)


class _BatchOnlyLoop(BaseLoop):

    def run(self,
            fn: Callable[..., Optional[Dict[str, Any]]],
            data_generator: Iterable[BatchArrays],
            metrics: Union[Sequence[str], type(ALL)] = NOT_SET,
            outputs: Union[Sequence[str], type(ALL)] = NOT_SET,
            aggregators: Optional[Mapping[str, BatchAggregator]] = None,
            excludes: Sequence[str] = ()
            ) -> Optional[Dict[str, Any]]:
        """
        Run the loop.

        Args:
            fn: The batch function to execute at each batch.
                The signature of `fn` should be ``(*arrays) -> None` or
                ``(*arrays) -> Dict[str, Any]``, which consumes the batch
                arrays produced by `data_generator`, and (maybe) returns the
                batch metrics or outputs.
            data_generator: Mini-batch data generator, yielding tuple of arrays.
            metrics: Names of metrics produced by `fn`.  These metrics will
                be aggregated by ``BatchAggregator('AVERAGE', axis=None)``,
                reported by ``self.logger``, and returned by this method.
                Defaults to ``SELF.RUN_BATCHES_DEFAULT_METRICS``.
            outputs: Names of outputs produced by `fn`.  These outputs will
                be aggregated by ``BatchAggregator('CONCAT', axis=0)``,
                and returned by this method.
                Defaults to ``SELF.RUN_BATCHES_DEFAULT_OUTPUTS``.
            aggregators: Dict from name to custom batch aggregators.
            excludes: The names to exclude, of items produced by `fn`.
                If a name is excluded, it will not be collected by any
                :class:`BatchAggregator`.
        """
        if not self._stage.is_active:
            with self:
                return self.run_batches(
                    fn, data_generator, metrics=metrics, outputs=outputs,
                    aggregators=aggregators, excludes=excludes
                )
        else:
            return self.run_batches(
                fn, data_generator, metrics=metrics, outputs=outputs,
                aggregators=aggregators, excludes=excludes
            )


class ValidationLoop(_BatchOnlyLoop):

    def __init__(self,
                 remote_doc: Optional[ExperimentDoc] = None,
                 callbacks: Sequence[Callback] = ()):
        super().__init__(
            stage=Stage(type=StageType.VALIDATION),
            remote_doc=remote_doc,
            callbacks=callbacks,
        )


class TestLoop(_BatchOnlyLoop):

    def __init__(self,
                 remote_doc: Optional[ExperimentDoc] = None,
                 callbacks: Sequence[Callback] = ()):
        super().__init__(
            stage=Stage(type=StageType.TEST),
            remote_doc=remote_doc,
            callbacks=callbacks,
        )


class PredictLoop(_BatchOnlyLoop):

    RUN_BATCHES_DEFAULT_METRICS = ()
    RUN_BATCHES_DEFAULT_OUTPUTS = ALL

    def __init__(self,
                 remote_doc: Optional[ExperimentDoc] = None,
                 callbacks: Sequence[Callback] = ()):
        super().__init__(
            stage=Stage(type=StageType.PREDICT),
            remote_doc=remote_doc,
            callbacks=callbacks,
        )


@dataclass
class DataGeneratorInfo(object):

    __slots__ = ('data_length', 'batch_size', 'batch_count')

    data_length: Optional[int]
    batch_size: Optional[int]
    batch_count: Optional[int]


def inspect_data_generator(g) -> Union[DataGeneratorInfo, Any]:
    if isinstance(g, DataStream):
        # since `DataStream` has all the interface of `DataGeneratorInfo`,
        # we just return it without constructing a new object
        return g
    return DataGeneratorInfo(data_length=None, batch_size=None,
                             batch_count=None)

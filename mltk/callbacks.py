import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import *

import numpy as np

from .checkpoint import BaseCheckpoint, CheckpointManager
from .formatting import MetricsFormatter, format_duration
from .metrics import ScalarMetricsLogger
from .mlstorage import ExperimentRemoteDoc
from .stateful import StatefulObjectGroup
from .utils import NOT_SET

__all__ = [
    'CallbackData', 'Callback',
    'LoggerCallback', 'AutoCheckpoint', 'EarlyStopping',
]


ProgressDict = MetricsDict = Dict[str, Any]


@dataclass
class CallbackData(object):
    """
    Data carried by a cycle begin/end event from :class:`Callback`.
    """

    __slots__ = ('stage', 'index', 'size', 'start_timestamp',
                 'end_timestamp', 'exc_time', 'metrics')

    stage: 'Stage'
    """The stage that calls the callback."""

    index: Optional[int]
    """Index of the epoch or batch."""

    size: Optional[int]
    """The size of the batch."""

    start_timestamp: float
    """Start timestamp of the stage/epoch/batch."""

    end_timestamp: Optional[float]
    """End timestamp of the stage/epoch/batch, available at the cycle end."""

    exc_time: Optional[float]
    """Execution time of the stage/epoch/batch, available at the cycle end."""

    metrics: Optional[MetricsDict]
    """Metrics dict, available at the cycle end."""


class Callback(object):
    """Base class of a callback for a machine learning stage."""

    ##################
    # general events #
    ##################
    def on_stage_begin(self, data: CallbackData):
        pass

    def on_stage_end(self, data: CallbackData):
        pass

    def on_epoch_begin(self, data: CallbackData):
        pass

    def on_epoch_end(self, data: CallbackData):
        pass

    def on_batch_begin(self, data: CallbackData):
        pass

    def on_batch_end(self, data: CallbackData):
        pass

    ################
    # train events #
    ################
    def on_train_begin(self, data: CallbackData):
        pass

    def on_train_end(self, data: CallbackData):
        pass

    def on_train_epoch_begin(self, data: CallbackData):
        pass

    def on_train_epoch_end(self, data: CallbackData):
        pass

    def on_train_batch_begin(self, data: CallbackData):
        pass

    def on_train_batch_end(self, data: CallbackData):
        pass

    #####################
    # validation events #
    #####################
    def on_validation_begin(self, data: CallbackData):
        pass

    def on_validation_end(self, data: CallbackData):
        pass

    def on_validation_batch_begin(self, data: CallbackData):
        pass

    def on_validation_batch_end(self, data: CallbackData):
        pass

    ###############
    # test events #
    ###############
    def on_test_begin(self, data: CallbackData):
        pass

    def on_test_end(self, data: CallbackData):
        pass

    def on_test_batch_begin(self, data: CallbackData):
        pass

    def on_test_batch_end(self, data: CallbackData):
        pass

    ##################
    # predict events #
    ##################
    def on_predict_begin(self, data: CallbackData):
        pass

    def on_predict_end(self, data: CallbackData):
        pass

    def on_predict_batch_begin(self, data: CallbackData):
        pass

    def on_predict_batch_end(self, data: CallbackData):
        pass


@dataclass
class _LoggerContext(object):
    """
    Class that maintains the context of an open stage in
    :class:`BaseLoggerCallback`.
    """

    __slots__ = ('stage', 'progress', 'metrics_collector', 'batch_metrics',
                 'last_console_log_time', 'last_remote_log_time')

    stage: 'Stage'
    progress: Dict[str, Any]
    metrics_collector: ScalarMetricsLogger
    """
    Metrics logger to accumulate the mean and std of metrics.  This logger
    will be cleared at the beginning when `on_epoch_begin` is called.

    For validation, test and predict, this should effectively accumulate
    the metrics throughout the whole stage, since the `on_epoch_begin`
    callback will never be called.
    """
    batch_metrics: Dict[str, Any]
    """
    The current batch metrics.  Will be cleared after each batch.
    """
    last_console_log_time: float
    """Last time that the logs have been written to console."""
    last_remote_log_time: float
    """Last time that the logs have been pushed to remote."""

    @staticmethod
    def new_context(stage) -> '_LoggerContext':
        now_time = time.time()
        return _LoggerContext(
            stage=stage,
            progress={},
            metrics_collector=ScalarMetricsLogger(),
            batch_metrics={},
            # set these two log times by the current time, such that these
            # logs will not be written immediately after the stage begins.
            last_console_log_time=now_time,
            last_remote_log_time=now_time,
        )

    def update_metrics(self,
                       metrics: Mapping[str, Any],
                       replace: bool = False,
                       batch_size: Optional[float] = None) -> None:
        """
        Update the epoch metrics logger and batch metrics dict (if a batch
        is currently active) according to `metrics`.

        Args:
            metrics: The batch, epoch or stage metrics from stage callback.
            replace: Whether or not to replace the epoch/stage metrics
                instead of updating them.
            batch_size: The batch size information from stage callback.
        """
        # We expect the metrics to be scalars.  If not, we shall take average.
        raw_metrics = {}
        averaged_metrics = {}
        if metrics:
            for key, val in metrics.items():
                key = self.stage.add_metric_prefix(key)
                if np.shape(val) == ():
                    averaged_metrics[key] = val
                else:
                    raw_metrics[key] = val

            updater = self.metrics_collector.replace \
                if replace else self.metrics_collector.update
            updater(raw_metrics)
            updater(averaged_metrics, weight=batch_size or 1.)

        # if inside a batch, update the batch metrics dict
        if self.stage.batch.is_active:
            self.batch_metrics.update(averaged_metrics)
            self.batch_metrics.update(
                {k: np.mean(v) for k, v in raw_metrics.items()})

    def copy_metrics_from_nested_context(self, ctx: '_LoggerContext'):
        """
        Copy the final metrics from nested stage.

        Args:
            ctx: The nested stage context.
        """
        # obtain the final metrics from the nested context
        nested_metrics = ctx.metrics_collector.to_json(mean_only=True)

        # if currently a batch is active, update the batch metrics
        if self.stage.batch.is_active:
            self.batch_metrics.update(nested_metrics)

        # update the final metrics
        self.metrics_collector.update(nested_metrics)

    def next_epoch(self):
        """Reset the internal states and enter the next epoch."""
        self.metrics_collector.clear()
        self.batch_metrics.clear()

    def next_batch(self):
        """Reset the internal states and enter the next batch."""
        self.batch_metrics.clear()


def _console_writer(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()


def _print_log(console_writer: Callable[[str], ...],
               text: str, nl: bool = True, show_time: bool = True):
    if console_writer is not None:
        if show_time:
            time_str = _format_datetime(datetime.now())
            text = f'[{time_str}] {text}'
        if nl:
            text += '\n'
        console_writer(text)


def _format_datetime(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')


class LoggerCallback(Callback):
    """
    Callback that logs training/testing/predicting progress and metrics
    to console and to MLStorage server.

    For performance considerations, batch metrics and progress information
    will be written to console for every ``console_log_interval`` seconds,
    and sent to server for every ``remote_log_interval`` seconds.

    The progress info will be stored as `progress.<stage.type>` field, and
    the batch metrics will be stored in `progress.<stage.type>.batch_metrics`
    field.  Stages with different types thus will not override the progress
    information and batch metrics of each other.

    Batch metrics will be accumulated by :class:`MetricsLogger`, and reported
    at the end of the epoch.  If the epoch callback provides metrics with the
    same names as the batch metrics, the epoch metrics will override the batch
    metrics.  These metrics are the epoch metrics.

    Moreover, metrics provided by the stage end callback are the stage metrics.
    Epoch metrics and stage metrics will be stored in the `result` field.
    For epoch and stage metrics of the train stage, the metrics will be saved
    as-is; but for other stages, the metrics names will be enforced to have
    the following prefix:

    *  validation stage: "val_" or "valid_".
    *  test stage: "test_"
    *  predict stage: "predict_"

    For nested stages (e.g., validation stage inside a train stage), the
    progress and metrics of the inner stages will not be written to the
    console, but will indeed be sent to the server.
    """

    ctx_stack: List[_LoggerContext]
    remote_doc: Optional[ExperimentRemoteDoc]
    console_writer: Optional[Callable[[str], None]]
    console_log_batch_freq: Optional[int]
    console_log_interval: Optional[float]
    remote_log_interval: Optional[float]
    enabled: bool

    def __init__(self,
                 remote_doc: Optional[ExperimentRemoteDoc] = NOT_SET,
                 metrics_formatter: MetricsFormatter = MetricsFormatter(),
                 console_writer: Optional[Callable[[str], None]] = _console_writer,
                 console_log_batch_freq: Optional[int] = None,
                 console_log_interval: Optional[float] = 10.,
                 remote_log_interval: Optional[float] = 60.):
        # complete NOT_SET arguments
        if remote_doc is NOT_SET:
            remote_doc = ExperimentRemoteDoc.from_env()
        if console_log_batch_freq is not None:
            console_log_interval = None

        self.ctx_stack = []
        self.remote_doc = remote_doc
        self.metrics_formatter = metrics_formatter
        self.console_writer = console_writer
        self.console_log_batch_freq = console_log_batch_freq
        self.console_log_interval = console_log_interval
        self.remote_log_interval = remote_log_interval

        self.enabled = (self.remote_doc is not None and
                        self.remote_log_interval is not None) or \
            self.console_log_batch_freq is not None or \
            self.console_log_interval is not None

    def _print_log(self, text: str, nl: bool = True, show_time: bool = False):
        if self.console_writer is not None:
            _print_log(self.console_writer, text=text, nl=nl,
                       show_time=show_time)

    @property
    def ctx(self) -> _LoggerContext:
        return self.ctx_stack[-1]

    @property
    def stage(self) -> 'Stage':
        return self.ctx.stage

    @property
    def is_nested(self) -> bool:
        return len(self.ctx_stack) > 1

    def _push_to_remote(self, result: Optional[Dict[str, Any]] = None):
        if self.remote_doc is not None:
            payload = {
                f'progress.{self.stage.name}': self.ctx.progress
            }
            if result:
                payload['result'] = result
            self.remote_doc.update(payload)
            self.ctx.last_remote_log_time = time.time()

    def _write_stage_or_epoch_end_logs(self,
                                       prefix: str = '',
                                       suffix: str = '',
                                       show_time: bool = False):
        # obtain the final results of the stage
        stage_result = self.ctx.metrics_collector.to_json()

        # now write the stage logs to console if the stage is not nested
        if not self.is_nested:
            buf = []
            if prefix:
                buf.append(prefix)
            if stage_result:
                result_str = self.metrics_formatter.format(
                    stage_result,
                    delimeters=(': ', ' - '),
                    known_names=self.stage.known_metrics
                )
                buf.append(result_str)
            if suffix:
                buf.append(suffix)
            self._print_log(' - '.join(buf), show_time=show_time)
            self.ctx.last_console_log_time = time.time()

        # push the stage logs to remote
        self._push_to_remote(stage_result)

    def _batch_console_head(self) -> str:
        # the batch counter
        total_batches = str(self.ctx.progress.get('total_batches', ''))
        batch = str(self.ctx.progress.get('batch', ''))
        if total_batches:
            return f'{batch:>{len(total_batches)}s}/{total_batches}'
        return batch

    def _update_progress_time_info(self, end_time: Optional[float]):
        # update elapsed
        if end_time is not None:
            self.ctx.progress['elapsed'] = end_time - self.stage.start_timestamp

        # update eta
        eta = self.stage.get_eta()
        if eta is not None and eta > 1e-7:
            self.ctx.progress['eta'] = eta
        else:
            self.ctx.progress.pop('eta', None)

    def on_stage_begin(self, data: CallbackData):
        self.ctx_stack.append(_LoggerContext.new_context(data.stage))
        if not self.is_nested:
            self._print_log(
                f'{self.stage.name.capitalize()} started',
                show_time=True
            )
            if self.remote_doc is not None:
                self.remote_doc.start_worker()

    def on_stage_end(self, data: CallbackData):
        try:
            # set the progress info
            self._update_progress_time_info(data.end_timestamp)

            # replace the epoch metrics with stage metrics, if provided
            if data.metrics:
                self.ctx.update_metrics(data.metrics, replace=True)

            # write the console logs and push to remote
            log_prefix = f'{self.stage.name.capitalize()} finished'
            log_suffix = ''
            if data.exc_time is not None:
                log_suffix = f'{self.stage.metric_prefix}time: ' \
                             f'{format_duration(data.exc_time, precision=3)}'
            self._write_stage_or_epoch_end_logs(
                log_prefix, log_suffix, show_time=True)

        finally:
            # pop this stage
            if len(self.ctx_stack) > 1:
                self.ctx_stack[-2].copy_metrics_from_nested_context(self.ctx)

            self.ctx_stack.pop()

            # stop the remote doc worker if there is no context left
            if not self.ctx_stack and self.remote_doc is not None:
                self.remote_doc.stop_worker()
                self.remote_doc.flush()

    def on_epoch_begin(self, data: CallbackData):
        # set the progress info
        self.ctx.progress['epoch'] = data.index + 1
        if data.stage.epoch.total is not None:
            self.ctx.progress['total_epochs'] = data.stage.epoch.total

        # set the context to enter next epoch
        self.ctx.next_epoch()

        # write epoch beginning log
        if not self.is_nested:
            self._print_log(f'>> Epoch {data.stage.epoch} <<', show_time=False)

    def on_epoch_end(self, data: CallbackData):
        # set the progress info
        self._update_progress_time_info(data.end_timestamp)
        if data.exc_time is not None:
            self.ctx.progress['epoch_time'] = data.exc_time

        # We use the metric values provided in `data.metrics` as the final
        # metric values for the epoch, to replace any batch metrics.
        self.ctx.update_metrics(data.metrics, replace=True)

        # write the console logs and push to remote
        log_prefix = self._batch_console_head()
        eta = self.stage.get_eta()
        if eta is not None:
            # just to be consistent with the format of batch logs
            log_prefix += f' - {format_duration(eta)}'

        log_suffix = ''
        if data.exc_time:
            log_suffix = f'epoch_time: ' \
                         f'{format_duration(data.exc_time, precision=3)}'
        self._write_stage_or_epoch_end_logs(log_prefix, log_suffix)

    def on_batch_begin(self, data: CallbackData):
        self.ctx.progress['batch'] = data.index + 1
        if data.stage.batch.total is not None:
            self.ctx.progress['total_batches'] = data.stage.batch.total
        self.ctx.progress.pop('batch_metrics', None)

        # set the context to enter next batch
        self.ctx.next_batch()

    def on_batch_end(self, data: CallbackData):
        # update the progress info
        self._update_progress_time_info(data.end_timestamp)
        if data.exc_time is not None:
            self.ctx.progress['batch_time'] = data.exc_time
        if data.metrics:
            # This assignment will be cleared at the beginning of the next batch
            self.ctx.progress['batch_metrics'] = data.metrics
        else:
            self.ctx.progress.pop('batch_metrics', None)

        # update the metrics
        self.ctx.update_metrics(data.metrics, batch_size=data.size)

        # check whether or not we need to write logs to console or to remote
        batch_id = data.index + 1
        need_remote_log = need_console_log = False

        if self.stage.epoch is not None and batch_id != self.stage.batch.total:
            # write to console only if this is not the final batch of an epoch
            # (where later the end of epoch log will be written).

            if not self.is_nested:
                # do not write to console for nested stage
                if self.console_log_batch_freq is not None:
                    # batch freq set, check the log freq
                    need_console_log = (
                        (batch_id % self.console_log_batch_freq) ==
                        0
                    )
                elif self.console_log_interval is not None:
                    # batch freq not set, check the log time interval
                    need_console_log = (
                        data.end_timestamp - self.ctx.last_console_log_time >=
                        self.console_log_interval
                    )

            if self.remote_log_interval is not None:
                need_remote_log = (
                    data.end_timestamp - self.ctx.last_remote_log_time >=
                    self.remote_log_interval
                )

        # obtain the results of the batch
        batch_result = self.ctx.batch_metrics

        # write logs to console
        if need_console_log:
            buf = [self._batch_console_head()]
            if 'eta' in self.ctx.progress:
                eta_str = format_duration(self.ctx.progress["eta"])
                buf.append(eta_str)
            if batch_result:
                result_str = self.metrics_formatter.format(
                    batch_result,
                    delimeters=(': ', ' - '),
                    known_names=self.stage.known_metrics,
                )
                buf.append(result_str)
            if data.exc_time is not None:
                buf.append(f'batch_time: '
                           f'{format_duration(data.exc_time, precision=3)}')
            self._print_log(' - '.join(buf), show_time=False)
            self.ctx.last_console_log_time = time.time()

        # push the logs to remote
        if need_remote_log:
            self._push_to_remote(batch_result)


class BaseCheckpointCallback(Callback):

    checkpoint: BaseCheckpoint
    root_dir: str
    stage: Optional['Stage'] = None
    checkpoint_manager: Optional[CheckpointManager] = None
    last_checkpoint_time: Optional[float] = None

    def __init__(self,
                 checkpoint: BaseCheckpoint,
                 root_dir: str):
        self.checkpoint = checkpoint
        self.root_dir = os.path.abspath(root_dir)

    def on_train_begin(self, data: CallbackData):
        if self.stage is not None:
            raise RuntimeError(f'`{self.__class__.__name__}` does not support '
                               f'nested train stage.')
        self.stage = data.stage
        self.checkpoint_manager = CheckpointManager(
            checkpoint=self.checkpoint,
            root_dir=self.root_dir,
            state_objects=StatefulObjectGroup({
                '__stage': data.stage.state_proxy()
            })
        )
        self.last_checkpoint_time = time.time()

    def on_train_end(self, data: CallbackData):
        self.stage = None
        self.checkpoint_manager = None
        self.last_checkpoint_time = None

    def make_checkpoint(self):
        epoch, batch = (self.stage.epoch.index + 1,
                        self.stage.batch.index + 1)
        ckpt_name = f'epoch-{epoch}-batch-{batch}'
        self.checkpoint_manager.save(ckpt_name)
        self.last_checkpoint_time = time.time()


class AutoCheckpoint(BaseCheckpointCallback):

    interval: Optional[float]
    epoch_freq: Optional[int]
    batch_freq: Optional[int]
    restore_checkpoint: Union[str, bool]

    def __init__(self,
                 checkpoint: BaseCheckpoint,
                 root_dir: str,
                 interval: Optional[float] = None,
                 epoch_freq: Optional[int] = None,
                 batch_freq: Optional[int] = None,
                 restore_checkpoint: Union[str, bool] = True):
        not_none_count = (
            int(interval is None) + int(epoch_freq is None) +
            int(batch_freq is None)
        )
        if not_none_count != 1:
            raise ValueError('One and only one of `interval`, `epoch_freq` '
                             'and `batch_freq` should be specified.')
        if not isinstance(restore_checkpoint, str) and \
                restore_checkpoint not in (True, False):
            raise TypeError(f'`restore_checkpoint` must be a str or a bool: '
                            f'got {restore_checkpoint!r}')

        super().__init__(checkpoint=checkpoint, root_dir=root_dir)
        self.interval = interval
        self.epoch_freq = epoch_freq
        self.batch_freq = batch_freq
        self.restore_checkpoint = restore_checkpoint

    def on_train_begin(self, data: CallbackData):
        super().on_train_begin(data)

        # restore the checkpoint
        if isinstance(self.restore_checkpoint, str):
            ckpt_path = self.restore_checkpoint
        elif self.restore_checkpoint is True:
            ckpt_path = self.checkpoint_manager.latest_checkpoint()
        else:
            ckpt_path = None

        if ckpt_path is not None:
            self.checkpoint_manager.restore(ckpt_path)
            _print_log(
                sys.stdout.write,
                f'restored from checkpoint: {ckpt_path}\n',
                show_time=True
            )
            sys.stdout.flush()

    def on_train_epoch_end(self, data: CallbackData):
        need_checkpoint = (
            (self.epoch_freq is not None and
             (data.index + 1) % self.epoch_freq == 0) or
            (self.interval is not None and
             data.end_timestamp - self.last_checkpoint_time >= self.interval)
        )
        if need_checkpoint:
            self.make_checkpoint()

    def on_train_batch_end(self, data: CallbackData):
        need_checkpoint = (
            (self.batch_freq is not None and
             (data.index + 1) % self.batch_freq == 0) or
            (data.index + 1 != self.stage.batch.total and
             # if the last batch in an epoch, better to make the checkpoint at
             # the end of the epoch
             self.interval is not None and
             data.end_timestamp - self.last_checkpoint_time >= self.interval)
        )
        if need_checkpoint:
            self.make_checkpoint()


class EarlyStopping(BaseCheckpointCallback):

    BEST_METRIC_VALUE_KEY = '__mltk.callbacks.EarlyStopping.best_metric_value'
    BEST_CHECKPOINT_NAME_KEY = '__mltk.callbacks.EarlyStopping.' \
                               'best_checkpoint_name'
    metric_name: str
    _metric_is_better: Callable[[Any, Any], bool]

    def __init__(self,
                 checkpoint: BaseCheckpoint,
                 root_dir: str,
                 metric_name: str,
                 metric_smaller_is_better: bool = True):
        super().__init__(checkpoint=checkpoint, root_dir=root_dir)
        self.metric_name = str(metric_name)
        if metric_smaller_is_better:
            self._metric_is_better = lambda new, old: old is None or new < old
        else:
            self._metric_is_better = lambda new, old: old is None or new > old

    @property
    def best_metric_value(self) -> Optional[Any]:
        return self.stage.memo.get(self.BEST_METRIC_VALUE_KEY, None)

    @best_metric_value.setter
    def best_metric_value(self, value: Any):
        self.stage.memo[self.BEST_METRIC_VALUE_KEY] = value

    @property
    def best_checkpoint_name(self) -> Optional[str]:
        return self.stage.memo.get(self.BEST_CHECKPOINT_NAME_KEY, None)

    @best_checkpoint_name.setter
    def best_checkpoint_name(self, value: str):
        self.stage.memo[self.BEST_CHECKPOINT_NAME_KEY] = value

    def _maybe_save_checkpoint(self, metrics: Optional[Dict[str, Any]]):
        if metrics and self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            if self._metric_is_better(metric_value, self.best_metric_value):
                ckpt_path = self.checkpoint_manager.save()
                ckpt_name = os.path.relpath(ckpt_path, self.root_dir)
                _print_log(
                    sys.stdout.write,
                    f'saved checkpoint for early-stopping: {ckpt_path}',
                    show_time=True
                )
                self.best_metric_value = metric_value
                self.best_checkpoint_name = ckpt_name

    def on_train_batch_end(self, data: CallbackData):
        super().on_train_batch_end(data)
        self._maybe_save_checkpoint(data.metrics)

    def on_train_epoch_end(self, data: CallbackData):
        super().on_train_epoch_end(data)
        self._maybe_save_checkpoint(data.metrics)

    def on_train_end(self, data: CallbackData):
        super().on_train_end(data)

        # restore from the best checkpoint, if any checkpoint is saved
        best_checkpoint_name = self.best_checkpoint_name
        if best_checkpoint_name is None:
            _print_log(
                sys.stdout.write,
                f'[WARNING] No checkpoint has been saved for early-stopping.  '
                f'Did you forget to update the validation metric '
                f'{self.metric_name!r}?',
                show_time=False
            )
        else:
            ckpt_path = os.path.join(self.root_dir, best_checkpoint_name)
            self.checkpoint_manager.restore(ckpt_path)
            _print_log(
                sys.stdout.write,
                f'restored early-stopping checkpoint from: '
                f'{ckpt_path}',
            )


# imported for type annotation on `Stage`
from .stage import Stage

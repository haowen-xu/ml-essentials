import sys
import time
from datetime import datetime

from dataclasses import dataclass
from typing import *

import numpy as np

from .formatting import MetricsFormatter, format_duration
from .metrics import ScalarMetricsLogger
from .mlstorage import ExperimentRemoteDoc
from .stage import *
from .utils import NOT_SET

__all__ = [
    'LoggerCallbackContext', 'LoggerCallback',
]


ProgressDict = MetricsDict = Dict[str, Any]


@dataclass
class LoggerCallbackContext(object):
    """
    Class that maintains the context of an open stage in
    :class:`BaseLoggerCallback`.
    """

    __slots__ = ('stage', 'progress', 'metrics_collector', 'batch_metrics',
                 'last_console_log_time', 'last_remote_log_time')

    stage: Stage
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
    def new_context(stage) -> 'LoggerCallbackContext':
        now_time = time.time()
        return LoggerCallbackContext(
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

    def copy_metrics_from_nested_context(self, ctx: 'LoggerCallbackContext'):
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


class LoggerCallback(StageCallback):
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

    ctx_stack: List[LoggerCallbackContext]
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

    @property
    def ctx(self) -> LoggerCallbackContext:
        return self.ctx_stack[-1]

    @property
    def stage(self) -> Stage:
        return self.ctx.stage

    @property
    def is_nested(self) -> bool:
        return len(self.ctx_stack) > 1

    def _print(self, text: str, nl: bool = True, time: bool = False):
        if self.console_writer is not None:
            if time:
                time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                text = f'[{time_str}] {text}'
            if nl:
                text += '\n'
            self.console_writer(text)

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
            self._print(' - '.join(buf), time=show_time)
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

    def on_stage_begin(self, data: StageCallbackData):
        self.ctx_stack.append(LoggerCallbackContext.new_context(data.stage))
        if not self.is_nested:
            self._print(f'{self.stage.name.capitalize()} started', time=True)
            self.remote_doc.start_worker()

    def on_stage_end(self, data: StageCallbackData):
        try:
            # set the progress info
            self._update_progress_time_info(self.stage.end_timestamp)

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
            if not self.ctx_stack:
                self.remote_doc.stop_worker()
                self.remote_doc.flush()

    def on_epoch_begin(self, data: StageCallbackData):
        # set the progress info
        self.ctx.progress['epoch'] = data.index + 1
        if data.stage.epoch.total is not None:
            self.ctx.progress['total_epochs'] = data.stage.epoch.total

        # set the context to enter next epoch
        self.ctx.next_epoch()

        # write epoch beginning log
        if not self.is_nested:
            self._print(f'>> Epoch {data.stage.epoch} <<')

    def on_epoch_end(self, data: StageCallbackData):
        # set the progress info
        self._update_progress_time_info(self.stage.epoch.end_timestamp)
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

    def on_batch_begin(self, data: StageCallbackData):
        self.ctx.progress['batch'] = data.index + 1
        if data.stage.batch.total is not None:
            self.ctx.progress['total_batches'] = data.stage.batch.total

        # set the context to enter next batch
        self.ctx.next_batch()

    def on_batch_end(self, data: StageCallbackData):
        # update the progress info
        self._update_progress_time_info(self.stage.batch.end_timestamp)
        if data.exc_time is not None:
            self.ctx.progress['batch_time'] = data.exc_time

        # update the metrics
        self.ctx.update_metrics(data.metrics, batch_size=data.size)

        # check whether or not we need to write console / remote log
        end_timestamp = self.stage.batch.end_timestamp

        # check whether or not we need to write logs to console or to remote
        batch_id = data.index + 1
        need_remote_log = need_console_log = False

        if self.stage.epoch is not None and batch_id != self.stage.batch.total:
            # write to console or push to remote only if this is not the final
            # batch of an epoch (where later the end of epoch log will be written).

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
                        end_timestamp - self.ctx.last_console_log_time >=
                        self.console_log_interval
                    )

            if self.remote_log_interval is not None:
                need_remote_log = (
                    end_timestamp - self.ctx.last_remote_log_time >=
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
            self._print(' - '.join(buf))
            self.ctx.last_console_log_time = time.time()

        # push the logs to remote
        if need_remote_log:
            self._push_to_remote(batch_result)

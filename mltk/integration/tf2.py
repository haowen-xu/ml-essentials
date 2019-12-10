from typing import *

import tensorflow as tf
from tensorflow import keras

from ..callbacks import *
from ..stage import *

__all__ = [
    'model_fit', 'model_evaluate', 'model_predict',
    'KerasCallbackWrapper',
]


if tf.__version__.split('.')[0] != '2':
    raise RuntimeError('`mltk.integration.tf2` requires TensorFlow 2.0')


CallbackTypes = Union[StageCallback, keras.callbacks.Callback]


def wrap_callbacks(callbacks: Sequence[CallbackTypes]
                   ) -> List[keras.callbacks.Callback]:
    ret = []
    stage_callbacks_buf = []

    for cb in (callbacks or ()):
        if isinstance(cb, StageCallback):
            stage_callbacks_buf.append(cb)
        else:
            if stage_callbacks_buf:
                ret.append(KerasCallbackWrapper(stage_callbacks_buf))
                stage_callbacks_buf.clear()
            ret.append(cb)

    if stage_callbacks_buf:
        ret.append(KerasCallbackWrapper(stage_callbacks_buf))

    return ret


def call_model_func(model, fn_,
                    *args,
                    console_log_batch_freq: Optional[int] = None,
                    console_log_interval: Optional[float] = 10.,
                    remote_log_interval: Optional[float] = 60.,
                    callbacks: Optional[Sequence[CallbackTypes]] = None,
                    verbose: int = 0,
                    **kwargs):
    # check the arguments
    callbacks = list(callbacks or ())
    logger_kwargs = {
        'console_log_batch_freq': console_log_batch_freq,
        'console_log_interval': console_log_interval,
        'remote_log_interval': remote_log_interval,
    }
    if verbose > 0:
        logger_kwargs['console_writer'] = None

    # create a new LoggerCallback if not exist
    if not any(isinstance(cb, LoggerCallback) for cb in callbacks):
        callbacks.append(LoggerCallback(**logger_kwargs))

    # now call `model.<fn_>`
    kwargs['callbacks'] = wrap_callbacks(callbacks)
    kwargs['verbose'] = verbose
    return getattr(model, fn_)(*args, **kwargs)


def model_fit(model,
              *args,
              console_log_batch_freq: Optional[int] = None,
              console_log_interval: Optional[float] = 10.,
              remote_log_interval: Optional[float] = 60.,
              callbacks: Optional[Sequence[CallbackTypes]] = None,
              verbose: int = 0,
              **kwargs):
    return call_model_func(
        model, 'fit',
        *args,
        console_log_batch_freq=console_log_batch_freq,
        console_log_interval=console_log_interval,
        remote_log_interval=remote_log_interval,
        callbacks=callbacks,
        verbose=verbose,
        **kwargs,
    )


def model_evaluate(model,
                   *args,
                   console_log_batch_freq: Optional[int] = None,
                   console_log_interval: Optional[float] = 10.,
                   remote_log_interval: Optional[float] = 60.,
                   callbacks: Optional[Sequence[CallbackTypes]] = None,
                   verbose: int = 0,
                   **kwargs):
    return call_model_func(
        model, 'evaluate',
        *args,
        console_log_batch_freq=console_log_batch_freq,
        console_log_interval=console_log_interval,
        remote_log_interval=remote_log_interval,
        callbacks=callbacks,
        verbose=verbose,
        **kwargs,
    )


def model_predict(model,
                  *args,
                  console_log_batch_freq: Optional[int] = None,
                  console_log_interval: Optional[float] = 10.,
                  remote_log_interval: Optional[float] = 60.,
                  callbacks: Optional[Sequence[CallbackTypes]] = None,
                  verbose: int = 0,
                  **kwargs):
    return call_model_func(
        model, 'predict',
        *args,
        console_log_batch_freq=console_log_batch_freq,
        console_log_interval=console_log_interval,
        remote_log_interval=remote_log_interval,
        callbacks=callbacks,
        verbose=verbose,
        **kwargs,
    )


def dict_get(logs: Optional[Dict[str, Any]], key: str, default: Any = None):
    if logs is None:
        return default
    return logs.get(key, default)


class KerasCallbackWrapper(keras.callbacks.Callback):

    callbacks: List[StageCallback]
    stage_stack: List[Stage]
    _metric_names: Tuple[str, ...] = None
    _metric_names_set: Set[str] = None

    def __init__(self, callbacks: Optional[Sequence[StageCallback]] = None):
        # check the Stage callbacks, and add a default LoggerCallback if
        # no such an instance is provided.
        callbacks = list(callbacks or ())
        if not any(isinstance(c, LoggerCallback) for c in callbacks):
            logger_callback = LoggerCallback(console_log_interval=None)
            if logger_callback.enabled:
                callbacks.append(logger_callback)

        # initialize the Keras callback
        super().__init__()
        self._chief_worker_only = True
        self.callbacks = callbacks
        self.stage_stack = []

    def set_params(self, params):
        super().set_params(params)

        # self.params.keys: [batch_size, epochs, steps, samples, verbose,
        #                    do_validation, metrics]
        params = self.params or {}
        self._metric_names = tuple(params.get('metrics') or ())
        self._metric_names_set = set(self._metric_names)

    @property
    def stage(self) -> Stage:
        return self.stage_stack[-1]

    def _filter_metrics(self, logs: Optional[Mapping[str, Any]]
                        ) -> Optional[Dict[str, Any]]:
        if logs is not None:
            return {k: v for k, v in logs.items()
                    if k in self._metric_names_set}

    #########
    # train #
    #########
    def on_train_begin(self, logs=None):
        self.stage_stack.append(
            Stage(
                type=StageType.TRAIN,
                total_epochs=dict_get(self.params, 'epochs'),
                total_batches=dict_get(self.params, 'steps'),
                batch_size=dict_get(self.params, 'batch_size'),
                data_count=dict_get(self.params, 'samples'),
                callbacks=self.callbacks,
                known_metrics=self._metric_names,
            )
        )
        self.stage.enter()

    def on_train_end(self, logs=None):
        self.stage.exit(self._filter_metrics(logs))
        self.stage_stack.pop()

    def on_epoch_begin(self, epoch, logs=None):
        self.stage.enter_epoch(epoch)

    def on_epoch_end(self, epoch, logs=None):
        # TODO: uncomment the following line once TF2.0 model metric aggregation
        #       is more reliable
        # self.stage.exit_epoch(self._filter_metrics(logs))
        self.stage.exit_epoch()

    def on_train_batch_begin(self, batch, logs=None):
        self.stage.enter_batch(batch, dict_get(logs, 'size'))

    def on_train_batch_end(self, batch, logs=None):
        self.stage.exit_batch(self._filter_metrics(logs))

    ########
    # test #
    ########
    def on_test_begin(self, logs=None):
        if len(self.stage_stack) > 0 and self.stage.type == StageType.TRAIN:
            self.stage_stack.append(
                Stage(
                    type=StageType.VALIDATION,
                    total_batches=dict_get(self.params, 'steps'),
                    batch_size=dict_get(self.params, 'batch_size'),
                    data_count=dict_get(self.params, 'samples'),
                    callbacks=self.callbacks,
                    known_metrics=self._metric_names,
                )
            )
        else:
            self.stage_stack.append(
                Stage(
                    type=StageType.TEST,
                    total_batches=dict_get(self.params, 'steps'),
                    batch_size=dict_get(self.params, 'batch_size'),
                    data_count=dict_get(self.params, 'samples'),
                    callbacks=self.callbacks,
                    known_metrics=self._metric_names,
                )
            )
        self.stage.enter()

    def on_test_end(self, logs=None):
        # TODO: uncomment the following line once TF2.0 model metric aggregation
        #       is more reliable
        # self.stage.exit(self._filter_metrics(logs))
        self.stage.exit()
        self.stage_stack.pop()

    def on_test_batch_begin(self, batch, logs=None):
        self.stage.enter_batch(batch, dict_get(logs, 'size'))

    def on_test_batch_end(self, batch, logs=None):
        self.stage.exit_batch(self._filter_metrics(logs))

    ###########
    # predict #
    ###########
    def on_predict_begin(self, logs=None):
        self.stage_stack.append(
            Stage(
                type=StageType.PREDICT,
                total_batches=dict_get(self.params, 'steps'),
                batch_size=dict_get(self.params, 'batch_size'),
                data_count=dict_get(self.params, 'samples'),
                callbacks=self.callbacks,
                known_metrics=self._metric_names,
            )
        )
        self.stage.enter()

    def on_predict_end(self, logs=None):
        # model.predict should not have any metrics to be reported.
        # self.stage.exit(self._filter_metrics(logs))
        self.stage.exit()
        self.stage_stack.pop()

    def on_predict_batch_begin(self, batch, logs=None):
        self.stage.enter_batch(batch, dict_get(logs, 'size'))

    def on_predict_batch_end(self, batch, logs=None):
        self.stage.exit_batch(self._filter_metrics(logs))

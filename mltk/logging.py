# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import Tuple, Union, Iterable, Dict, Optional, Callable, Any

import numpy as np

from .events import EventHost, Event
from .utils import format_duration, DocInherit

__all__ = ['StatisticsCollector', 'MetricLogger']

MetricType = Union[int, float, np.ndarray]
MetricSortKeyType = Callable[[str], Any]
MetricFormatterType = Callable[[str, MetricType], Any]
MetricsCollectedCallbackType = Callable[[Dict[str, MetricType]], None]


class StatisticsCollector(object):
    """
    Class to compute :math:`\\mathrm{E}[X]` and :math:`\\operatorname{Var}[X]`.

    To collect statistics of a scalar:

    >>> collector = StatisticsCollector()
    >>> for value in [1., 2., 3., 4.]:
    ...     collector.collect(value)
    >>> collector.mean, collector.var, collector.stddev
    (array(2.5), array(1.25), array(1.11803399))
    >>> collector.collect(np.array([5., 6., 7., 8.]))
    >>> collector.mean, collector.var, collector.stddev
    (array(4.5), array(5.25), array(2.29128785))

    weighted statistics:

    >>> collector = StatisticsCollector()
    >>> for value in [1., 2., 3., 4.]:
    ...     collector.collect(value, weight=value)
    >>> collector.weight_sum
    10.0
    >>> collector.mean, collector.var, collector.stddev
    (array(3.), array(1.), array(1.))
    >>> collector.collect(np.array([5., 6., 7., 8.]),
    ...                   weight=np.array([5., 6., 7., 8.]))
    >>> collector.weight_sum
    36.0
    >>> collector.mean, collector.var, collector.stddev
    (array(5.66666667), array(3.88888889), array(1.97202659))

    To collect element-wise statistics of a vector:

    >>> collector = StatisticsCollector(shape=[3])
    >>> x = np.arange(12).reshape([4, 3])
    >>> for value in x:
    ...     collector.collect(value)
    >>> collector.mean
    array([4.5, 5.5, 6.5])
    >>> collector.var
    array([11.25, 11.25, 11.25])
    >>> collector.stddev
    array([3.35410197, 3.35410197, 3.35410197])
    """

    def __init__(self, shape: Iterable[int] = (),
                 dtype=np.float64):
        """
        Construct the :class:`StatisticsCollector`.

        Args:
            shape: Shape of the values. The statistics will be collected for
                per element of the values. (default is ``()``).
            dtype: Data type of the statistics.
        """
        self._shape = tuple(shape)
        self._dtype = np.dtype(dtype)
        self._mean = np.zeros(shape=shape, dtype=self.dtype)      # E[X]
        self._square = np.zeros(shape=shape, dtype=self.dtype)    # E[X^2]
        self._counter = 0
        self._weight_sum = 0.

    def reset(self):
        """Reset the collector to initial state."""
        self._mean = np.zeros(shape=self._shape, dtype=self.dtype)
        self._square = np.zeros(shape=self._shape, dtype=self.dtype)
        self._counter = 0
        self._weight_sum = 0.

    @property
    def shape(self) -> Tuple[int]:
        """Get the shape of the values."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Get the data type of the statistics."""
        return self._dtype

    @property
    def mean(self) -> np.ndarray:
        """Get the mean of the values, i.e., :math:`\\mathrm{E}[X]`."""
        return self._mean

    @property
    def square(self) -> np.ndarray:
        """Get :math:`\\mathrm{E}[X^2]` of the values."""
        return self._square

    @property
    def var(self) -> np.ndarray:
        """
        Get the variance of the values, i.e., :math:`\\operatorname{Var}[X]`.
        """
        return np.asarray(np.maximum(self._square - self._mean ** 2, 0.))

    @property
    def stddev(self) -> np.ndarray:
        """
        Get the std of the values, i.e., :math:`\\sqrt{\\operatorname{Var}[X]}`.
        """
        return np.asarray(np.sqrt(self.var))

    @property
    def weight_sum(self) -> float:
        """Get the weight summation."""
        return self._weight_sum

    @property
    def has_value(self) -> bool:
        """Whether or not any value has been collected?"""
        return self._counter > 0

    @property
    def counter(self) -> int:
        """Get the counter of collected values."""
        return self._counter

    def collect(self, values: Union[float, np.ndarray],
                weight: Union[float, np.ndarray] = 1.):
        """
        Update the statistics from values.

        This method uses the following equation to update `mean` and `square`:

        .. math::
            \\frac{\\sum_{i=1}^n w_i f(x_i)}{\\sum_{j=1}^n w_j} =
                \\frac{\\sum_{i=1}^m w_i f(x_i)}{\\sum_{j=1}^m w_j} +
                \\frac{\\sum_{i=m+1}^n w_i}{\\sum_{j=1}^n w_j} \\Bigg(
                    \\frac{\\sum_{i=m+1}^n w_i f(x_i)}{\\sum_{j=m+1}^n w_j} -
                    \\frac{\\sum_{i=1}^m w_i f(x_i)}{\\sum_{j=1}^m w_j}
                \\Bigg)

        Args:
            values: Values to be collected in batch, numpy array or scalar
                whose shape ends with ``self.shape``. The leading shape in
                front of ``self.shape`` is regarded as the batch shape.
            weight: Weights of the `values`, should be broadcastable against
                the batch shape. (default is 1)

        Raises:
            ValueError: If the shape of `values` does not end with `self.shape`.
        """
        values = np.asarray(values)
        if not values.size:
            return
        weight = np.asarray(weight)
        if not weight.size:
            weight = np.asarray(1.)

        if self._shape:
            if values.shape[-len(self._shape):] != self._shape:
                raise ValueError(
                    'Shape mismatch: {} not ending with {}'.format(
                        values.shape, self._shape
                    )
                )
            batch_shape = values.shape[:-len(self._shape)]
        else:
            batch_shape = values.shape

        batch_weight = np.ones(shape=batch_shape, dtype=np.float) * weight
        batch_weight = np.reshape(batch_weight,
                                  batch_weight.shape + (1,) * len(self._shape))
        batch_weight_sum = np.sum(batch_weight)
        normed_batch_weight = batch_weight / batch_weight_sum

        self._weight_sum += batch_weight_sum
        discount = batch_weight_sum / self._weight_sum

        def update_array(arr, update):
            reduce_axis = tuple(range(len(batch_shape)))
            update_reduced = normed_batch_weight * update
            if reduce_axis:
                update_reduced = np.sum(update_reduced, axis=reduce_axis)
            arr += discount * (update_reduced - arr)
        update_array(self._mean, values)
        update_array(self._square, values ** 2)
        self._counter += batch_weight.size


def metrics_sort_key(key: str):
    key_suffix = key.lower().rsplit('_', 1)[-1]
    if key_suffix in ('time', 'timer'):
        return -1, key
    else:
        return 0, key


def metrics_formatter(key: str, value: MetricType) -> str:
    key_suffix = key.lower().rsplit('_', 1)[-1]
    if key_suffix in ('time', 'timer'):
        return format_duration(value)
    else:
        return f'{float(value):.6g}'


class MetricLogger(object):
    """
    Class to log training and evaluation metrics.

    The metric statistics during a certain period can be obtained by
    :meth:`format_logs()`.  Note that each time :meth:`format_logs()`
    is called, the statistics will be cleared, unless calling
    `format_logs(clear_stats=False)`.

    >>> logger = MetricLogger()
    >>> logger.collect(train_loss=5, train_acc=90, train_time=30)
    >>> logger.collect(train_loss=6)
    >>> logger.format_logs()  # note format_logs will clear the stats
    'train_time: 30s; train_acc: 90; train_loss: 5.5 (±0.5)'
    >>> logger.collect(train_loss=5)
    >>> logger.format_logs()
    'train_loss: 5'

    It is possible to get notified when metrics are collected.  For example:

    >>> logger = MetricLogger()
    >>> logger.on_metrics_collected.do(lambda d: print('callback', d))
    >>> logger.collect({'train_loss': 5}, train_acc=90)
    callback {'train_loss': 5, 'train_acc': 90}
    """

    def __init__(self, sort_key: MetricSortKeyType = metrics_sort_key,
                 formatter: MetricFormatterType = metrics_formatter):
        """
        Construct a new :class:`MetricLogger`.

        Args:
            sort_key: The function to get the key for sorting metrics.
                For example, the following sort_key function will place
                all "time" metrics (e.g., "train_time") at first,
                all "loss" metrics (e.g., "train_loss") the second,
                and all other metrics at last.

                >>> def my_sort_key(key: str):
                ...     key_suffix = key.lower().rsplit('_', 1)[-1]
                ...     if key_suffix == 'time':
                ...         return -2, key
                ...     elif key_suffix == 'loss':
                ...         return -1, key
                ...     else:
                ...         return 0, key

                >>> logger = MetricLogger(sort_key=my_sort_key)
                >>> logger.collect(train_loss=5, train_acc=90, train_time=30)
                >>> logger.format_logs()
                'train_time: 30s; train_loss: 5; train_acc: 90'

            formatter: The function to format the value of a metric.
                It should support formatting both the mean and std.
                For example:

                >>> def my_formatter(key: str, value: MetricType) -> str:
                ...     key_suffix = key.rsplit('_')[-1]
                ...     if key_suffix == 'time':
                ...         return format_duration(value, short_units=False)
                ...     elif key_suffix == 'loss':
                ...         return f'{value:.6f}'
                ...     else:
                ...         return f'{value:.6g}'

                >>> logger = MetricLogger(formatter=my_formatter)
                >>> logger.collect(train_loss=5, train_acc=90, train_time=30)
                >>> logger.format_logs()
                'train_time: 30 seconds; train_acc: 90; train_loss: 5.000000'
        """
        self._metrics_sort_key = sort_key
        self._metrics_formatter = formatter

        # dict to record the metric statistics of current batch
        self._stats_collectors = defaultdict(StatisticsCollector)
        # event host and events
        self._events = EventHost()
        self._on_metrics_collected = self.events['metrics_collected']

    @property
    def stats_collectors(self) -> Dict[str, StatisticsCollector]:
        return self._stats_collectors

    @property
    def events(self) -> EventHost:
        """Get the event host."""
        return self._events

    @property
    def on_metrics_collected(self) -> Event:
        """Get the metrics collected event."""
        return self._on_metrics_collected

    def clear_stats(self):
        """Clear the metrics statistics."""
        for v in self._stats_collectors.values():
            v.reset()

    def collect(self, metrics: Optional[Dict[str, MetricType]] = None,
                **kwargs: MetricType):
        """
        Collect the metrics.

        Metrics can be specified as dict via the first positional argument,
        or via named arguments.  For example:

        >>> logger = MetricLogger()
        >>> logger.collect(train_loss=5)
        >>> logger.collect({'train_loss': 6, 'train_acc': 90})
        >>> logger.format_logs()
        'train_acc: 90; train_loss: 5.5 (±0.5)'

        If a metric appears in both the first positional argument and the
        named arguments, then the value from the named arguments will
        override the value from the positional argument.  For example:

        >>> logger = MetricLogger()
        >>> logger.collect({'train_loss': 5}, train_loss=6)
        >>> logger.format_logs()
        'train_loss: 6'

        Calling :meth:`collect()` without any metric will cause no effect
        (in particular, :obj:`metrics_updated` event will not be fired).

        >>> logger = MetricLogger()
        >>> logger.on_metrics_collected.do(lambda d: print('callback', d))
        >>> logger.collect(train_loss=5)
        callback {'train_loss': 5}
        >>> logger.collect()

        Args:
            metrics: The metrics dict.
            \\**kwargs: The metrics.
        """
        if metrics and kwargs:
            merged = {}
            merged.update(metrics)
            merged.update(kwargs)
        elif metrics:
            merged = metrics
        elif kwargs:
            merged = kwargs
        else:
            return

        for key, value in merged.items():
            self._stats_collectors[key].collect(value)
        self.on_metrics_collected.fire(merged)

    def format_logs(self, clear_stats: bool = True) -> str:
        """
        Format the metric statistics as log.

        The metric statistics will be cleared after :meth:`format_logs()`
        is called, unless `clear_stats` is set to :obj:`False`.

        >>> logger = MetricLogger()
        >>> logger.collect(train_loss=5)
        >>> logger.format_logs()
        'train_loss: 5'
        >>> logger.collect(train_loss=6)
        >>> logger.format_logs(clear_stats=False)
        'train_loss: 6'
        >>> logger.collect(train_loss=5)
        >>> logger.format_logs()
        'train_loss: 5.5 (±0.5)'

        Args:
            clear_stats: Whether or not to clear the statistics after
                formatting the log? (default :obj:`True`)

        Returns:
            The formatted log.
        """
        buf = []
        for key in sorted(self._stats_collectors.keys(),
                          key=self._metrics_sort_key):
            metric_stat = self._stats_collectors[key]
            if metric_stat.has_value:
                mean = self._metrics_formatter(key, metric_stat.mean)
                if metric_stat.counter > 1:
                    std = self._metrics_formatter(key, metric_stat.stddev)
                    std = f' (±{std})'
                else:
                    std = ''
                buf.append(f'{key}: {mean}{std}')
        if clear_stats:
            self.clear_stats()
        return '; '.join(buf)

from typing import Tuple, Union, Iterable

import numpy as np

__all__ = ['StatisticsCollector']


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

import os
from typing import *

import torch

from ..checkpoint import BaseCheckpoint
from ..stateful import StatefulObject

__all__ = ['TorchCheckpoint']


class _TorchCheckpointableObject(StatefulObject):
    """Wraps a PyTorch checkpointable object into :class:`StatefulObject`."""

    def __init__(self, torch_object):
        self.torch_object = torch_object

    def get_state_dict(self) -> Dict[str, Any]:
        return self.torch_object.state_dict()

    def set_state_dict(self, state: Dict[str, Any]):
        self.torch_object.load_state_dict(state)


class TorchCheckpoint(BaseCheckpoint):
    """
    A checkpoint object that supports to save and recover PyTorch checkpointable
    objects (i.e., objects with method :meth:`state_dict()` and
    :meth:`load_state_dict()`).

    Usage::

        # create the PyTorch objects
        class Net(torch.nn.Module):
            ...

        net = Net(...)
        optimizer = torch.optim.Adam(...)

        # construct the checkpoint object
        checkpoint = TorchCheckpoint(net=net, optimizer=optimizer)

        # save a checkpoint
        checkpoint.save(checkpoint_path)

        # restore the checkpoint
        checkpoint.restore(checkpoint_path)
    """

    def __init__(self, **torch_objects: Any):
        def to_stateful_object(obj) -> StatefulObject:
            if isinstance(obj, StatefulObject):
                return obj
            elif hasattr(obj, 'state_dict') and hasattr(obj, 'load_state_dict'):
                return _TorchCheckpointableObject(obj)
            else:
                raise TypeError(
                    f'Object must be a :class:`StatefulObject`, or has '
                    f'`state_dict()` and `load_state_dict()` methods: '
                    f'got {obj!r}'
                )

        self._objects: Dict[str, StatefulObject] = {
            k: to_stateful_object(o)
            for k, o in torch_objects.items()
        }

    def _restore(self, checkpoint_path: str) -> None:
        data_path = os.path.join(checkpoint_path, 'data.pth')
        state_dict = torch.load(data_path)

        # check whether or not all keys exist
        for k in self._objects:
            if k not in state_dict:
                raise ValueError(f'Key {k!r} does not exist in the '
                                 f'state dict recovered from: {data_path}')

        # load the state dict
        for k, o in self._objects.items():
            o.set_state_dict(state_dict[k])

    def _save(self, checkpoint_path: str) -> None:
        # generate the state dict
        state_dict = {
            k: o.get_state_dict()
            for k, o in self._objects.items()
        }

        # save the objects
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        data_path = os.path.join(checkpoint_path, 'data.pth')
        torch.save(state_dict, data_path)

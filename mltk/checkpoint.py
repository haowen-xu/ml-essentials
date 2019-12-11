import codecs
import json
import os
import shutil
import time
from datetime import datetime
from logging import getLogger
from typing import *

from .stateful import StatefulObject, StatefulObjectGroup, StateSaver

__all__ = ['BaseCheckpoint', 'CheckpointManager']

StatefulObjects = Union[StatefulObject, Dict[str, StatefulObject]]


class BaseCheckpoint(object):
    """
    Base interface of a checkpoint object.

    Any attribute attached to a checkpoint object should be saved via
    :meth:`save()`, and restored via :meth:`restore()`.

    The true checkpoint classes for specific backends should be implemented
    in the modules under the package ``mltk.integration``.
    """

    def _save(self, checkpoint_path: str) -> None:
        raise NotImplementedError()

    def save(self,
             checkpoint_dir: str,
             state_objects: Optional[StatefulObjects] = None,
             overwrite: bool = False) -> None:
        """
        Save checkpoint to `checkpoint_dir`.

        Args:
            checkpoint_dir: The directory where to save the checkpoint.
            state_objects: Additional stateful object(s) to be saved,
                alongside the backend checkpoint file.
            overwrite: Whether or not to overwrite exist checkpoint?
        """
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        if state_objects is not None and \
                not isinstance(state_objects, StatefulObject):
            state_objects = StatefulObjectGroup(state_objects)

        # check whether or not we shall overwrite existing file/directory
        if os.path.exists(checkpoint_dir):
            if not overwrite:
                raise IOError(f'`checkpoint_dir` already exists: '
                              f'{checkpoint_dir}')
            if os.path.isdir(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            else:
                os.remove(checkpoint_dir)

        # now save the checkpoint and state objects
        os.makedirs(checkpoint_dir, exist_ok=True)
        state_path = os.path.join(checkpoint_dir, 'state.npz')
        ckpt_path = os.path.join(checkpoint_dir, 'ckpt')

        if state_objects is not None:
            StateSaver(state_objects).save(state_path)
        self._save(ckpt_path)

    def _restore(self, checkpoint_path: str) -> None:
        raise NotImplementedError()

    def restore(self,
                checkpoint_dir: str,
                state_objects: Optional[StatefulObjects] = None) -> None:
        """
        Restore checkpoint from `checkpoint_dir`.

        Args:
            checkpoint_dir: The directory where the checkpoint was saved.
            state_objects: Additional stateful objects to be restored,
                alongside the backend checkpoint file.
        """
        # backup the original object state
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        if state_objects is not None and \
                not isinstance(state_objects, StatefulObject):
            state_objects = StatefulObjectGroup(state_objects)

        original_state = state_objects.get_state_dict() \
            if state_objects is not None else None

        # check whether the checkpoint exists
        state_path = os.path.join(checkpoint_dir, 'state.npz')
        ckpt_path = os.path.join(checkpoint_dir, 'ckpt')

        if not os.path.exists(ckpt_path) or \
                (state_objects is not None and not os.path.isfile(state_path)):
            raise IOError(f'Checkpoint does not exist or is malformed: '
                          f'{checkpoint_dir}')

        # load the state object and checkpoint
        try:
            if state_objects is not None:
                StateSaver(state_objects).load(state_path)
            self._restore(ckpt_path)
        except:
            if state_objects is not None:
                state_objects.set_state_dict(original_state)
            raise


class CheckpointManager(object):

    checkpoint: BaseCheckpoint
    root_dir: str
    state_objects: Optional[StatefulObjects]
    checkpoint_index_file: str
    max_to_keep: Optional[int]
    _latest_checkpoint: Optional[str]
    _checkpoint_list: List[str]

    def __init__(self,
                 checkpoint: BaseCheckpoint,
                 root_dir: str,
                 state_objects: Optional[StatefulObjects],
                 checkpoint_index_file: str = 'checkpoint.json',
                 max_to_keep: Optional[int] = None):
        root_dir = os.path.abspath(root_dir)
        self.checkpoint = checkpoint
        self.root_dir = root_dir
        self.state_objects = state_objects
        self.checkpoint_index_file = checkpoint_index_file
        self.max_to_keep = max_to_keep

        # load the checkpoint index file
        index_path = os.path.join(root_dir, checkpoint_index_file)
        if os.path.isfile(index_path):
            with codecs.open(index_path, 'rb', 'utf-8') as f:
                cnt = f.read()
            index_content = json.loads(cnt)
        else:
            index_content = {}

        self._latest_checkpoint = index_content.get('latest_checkpoint', None)
        self._checkpoint_list = list(index_content.get('checkpoint_list', []))

    def _save_index_file(self):
        index_path = os.path.join(self.root_dir, self.checkpoint_index_file)
        cnt = json.dumps({
            'latest_checkpoint': self._latest_checkpoint,
            'checkpoint_list': self._checkpoint_list,
        })
        with codecs.open(index_path, 'wb', 'utf-8') as f:
            f.write(cnt)

    def latest_checkpoint(self) -> Optional[str]:
        if self._latest_checkpoint is not None:
            return os.path.join(self.root_dir, self._latest_checkpoint)

    def save(self, name: Optional[str] = None) -> str:
        # get a unique checkpoint name
        if name is None:
            name = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            if name in self._checkpoint_list:
                name = datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')
            while name in self._checkpoint_list:
                time.sleep(0.01)
                name = datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')
        else:
            i = 0
            base_name = name
            while name in self._checkpoint_list:
                i += 1
                name = f'{base_name}_{i}'

        # now save the checkpoint and index file
        path = os.path.join(self.root_dir, name)
        names_to_purge = []

        try:
            # save checkpoint and update index file
            self.checkpoint.save(path, self.state_objects, overwrite=True)
            self._latest_checkpoint = name
            self._checkpoint_list.append(name)

            # purge old checkpoint if `max_to_keep` is configured
            if self.max_to_keep is not None:
                while len(self._checkpoint_list) > self.max_to_keep:
                    names_to_purge.append(self._checkpoint_list.pop(0))

            # save the new index file
            self._save_index_file()
        except:
            shutil.rmtree(path)
            raise

        # checkpoint saved, purge old checkpoint
        if names_to_purge is not None:
            for old_name in names_to_purge:
                old_path = os.path.join(self.root_dir, old_name)
                try:
                    if os.path.exists(old_path):
                        shutil.rmtree(old_path)
                except Exception:  # pragma: no cover
                    getLogger(__name__).warning(
                        'Failed to purge old checkpoint: %s', old_path)

        return path

    def restore(self, path: str):
        self.checkpoint.restore(path, self.state_objects)

    def restore_latest(self, ignore_not_exist: bool = True):
        latest_checkpoint = self.latest_checkpoint()
        if latest_checkpoint is not None:
            self.restore(latest_checkpoint)
        elif not ignore_not_exist:
            raise IOError('No checkpoint can be restored.')

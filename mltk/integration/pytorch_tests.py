import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from mltk import SimpleStatefulObject
from mltk.integration.pytorch import _TorchCheckpointableObject, TorchCheckpoint


class TorchCheckpointableObjectTestCase(unittest.TestCase):

    def test_save_restore(self):
        x = torch.from_numpy(np.random.normal(size=[2, 5]).astype(np.float32))

        # save from one layer
        layer = torch.nn.Linear(5, 3)
        o = _TorchCheckpointableObject(layer)
        expected_state = layer.state_dict()
        target_state = o.get_state_dict()
        self.assertEqual(set(target_state), set(expected_state))
        for k in target_state:
            self.assertTrue(torch.allclose(target_state[k], expected_state[k]))

        # restore to another layer
        expected_out = layer(x)
        layer2 = torch.nn.Linear(5, 3)
        out2 = layer2(x)
        self.assertFalse(torch.allclose(out2, expected_out))

        o2 = _TorchCheckpointableObject(layer2)
        o2.set_state_dict(target_state)
        out2 = layer2(x)
        self.assertTrue(torch.allclose(out2, expected_out))


class TorchCheckpointTestCase(unittest.TestCase):

    def test_invalid_type(self):
        with pytest.raises(TypeError,
                           match=r'Object must be a :class:`StatefulObject`, '
                                 r'or has `state_dict\(\)` and '
                                 r'`load_state_dict\(\)` methods: got 123'):
            _ = TorchCheckpoint(obj=123)

    def test_save_restore(self):
        x = torch.from_numpy(np.random.normal(size=[2, 5]).astype(np.float32))

        with TemporaryDirectory() as temp_dir:
            root_dir = os.path.join(temp_dir, 'ckpt')

            # test save
            layer = torch.nn.Linear(5, 3)
            obj = SimpleStatefulObject()
            obj.value = 123456
            ckpt = TorchCheckpoint(obj=obj, layer=layer)
            ckpt.save(root_dir)

            # test restore
            layer2 = torch.nn.Linear(5, 3)
            obj2 = SimpleStatefulObject()
            ckpt2 = TorchCheckpoint(obj=obj2, layer=layer2)
            ckpt2.restore(root_dir)

            # compare two objects
            out = layer(x)
            out2 = layer2(x)
            self.assertTrue(torch.allclose(out2, out))
            self.assertEqual(obj2.value, 123456)

            # test partial restore
            layer3 = torch.nn.Linear(5, 3)
            ckpt3 = TorchCheckpoint(layer=layer3)
            ckpt3.restore(root_dir)
            self.assertTrue(torch.allclose(layer3(x), out))

            # test restore error
            ckpt4 = TorchCheckpoint(layer=layer3, xyz=SimpleStatefulObject())
            with pytest.raises(ValueError,
                               match=f'Key \'xyz\' does not exist in '
                                     f'the state dict recovered from: '
                                     f'{root_dir}'):
                ckpt4.restore(root_dir)



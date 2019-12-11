import os
import unittest

if os.environ['KERAS_BACKEND'] == 'keras':
    import keras
elif os.environ['KERAS_BACKEND'] == 'tensorflow':
    from tensorflow import keras
else:
    raise RuntimeError('The keras backend is not specified.')


class KerasCallbackWrapperTestCase(unittest.TestCase):

    def test_sample(self):
        self.assertEqual(1, 1)


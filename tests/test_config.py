import codecs
import os
import unittest
from tempfile import TemporaryDirectory

import pytest

from ml_essentials.config import (is_config_attribute, Config,
                                  ConfigField, BoolValidator, ValidationContext,
                                  ConfigValidator, StrValidator, FloatValidator,
                                  IntValidator, ConfigValidationError,
                                  get_validator, ConfigLoader)


class ConfigTestCase(unittest.TestCase):

    def test_is_config_attribute(self):
        class SubConfig(Config):
            value = 1
            _private = 2
            field = ConfigField(str)

            class nested(Config):
                value = 123

            nested2 = Config(value=456)

            def get_value(self):
                return self.value

            @property
            def the_value(self):
                return self.value

            @classmethod
            def class_value(cls):
                return cls.value

            @staticmethod
            def static_value():
                return SubConfig.value

        c = SubConfig()
        c.value2 = 2
        for key in ['value', 'field', 'nested', 'nested2']:
            self.assertTrue(is_config_attribute(SubConfig, key))

        for key in ['value', 'value2', 'field', 'nested', 'nested2']:
            self.assertTrue(is_config_attribute(c, key))

        for key in ['value3', '_private', 'get_value', 'the_value',
                    'class_value', 'static_value']:
            self.assertFalse(is_config_attribute(SubConfig, key))
            self.assertFalse(is_config_attribute(c, key))

        for key in ['value3']:
            self.assertTrue(is_config_attribute(
                SubConfig, key, require_existence=False))
            self.assertTrue(is_config_attribute(
                c, key, require_existence=False))

    def test_ConfigField(self):
        # not specifying type and default
        field = ConfigField()
        self.assertEqual(repr(field), 'ConfigField(nullable=True)')

        # specifying default value but not type
        field = ConfigField(default=123)
        self.assertIsNone(field.type)
        self.assertEqual(
            repr(field), 'ConfigField(default=123, nullable=True)')

        # specifying type but not default value
        class MyConfig(Config):
            a = 123

        field = ConfigField(MyConfig)
        self.assertIs(field.type, MyConfig)
        self.assertEqual(
            repr(field),
            'ConfigField(type=ConfigTestCase.test_ConfigField.'
            '<locals>.MyConfig, nullable=True)'
        )

        # specifying the description
        field = ConfigField(int, description='hello')
        self.assertEqual(field.description, 'hello')
        self.assertEqual(
            repr(field), 'ConfigField(type=int, nullable=True)')

        # specifying nullable
        field = ConfigField(nullable=False)
        self.assertFalse(field.nullable)
        self.assertEqual(repr(field), 'ConfigField(nullable=False)')

        # specifying the choices
        field = ConfigField(int, choices=[1, 2, 3])
        self.assertIsInstance(field.choices, tuple)
        self.assertTupleEqual(field.choices, (1, 2, 3))
        self.assertEqual(
            repr(field),
            'ConfigField(type=int, nullable=True, choices=[1, 2, 3])'
        )

    def test_Config_setattr(self):
        config = Config()
        with pytest.raises(TypeError,
                           match='`value` must not be a ConfigField'):
            config.value = ConfigField(int)

        with pytest.raises(AttributeError,
                           match='`name` must not contain \'.\': \'a.b\''):
            setattr(config, 'a.b', 123)


class ConfigLoaderTestCase(unittest.TestCase):

    def test_construction(self):
        class MyConfig(Config):
            pass

        loader = ConfigLoader(config_cls=MyConfig)
        self.assertIs(loader.config_cls, MyConfig)
        self.assertFalse(loader.validate_all)

        loader = ConfigLoader(config_cls=MyConfig, validate_all=True)
        self.assertTrue(loader.validate_all)

        with pytest.raises(TypeError,
                           match='`config_cls` is not Config or a subclass of '
                                 'Config: <class \'str\'>'):
            _ = ConfigLoader(str)

    def test_load_object(self):
        class MyConfig(Config):
            class nested1(Config):
                a = 123
                b = ConfigField(float, default=None)

            class nested2(Config):
                c = 789


        # test feed object of invalid type
        loader = ConfigLoader(MyConfig)
        with pytest.raises(TypeError,
                           match='`key_values` must be a dict or a Config '
                                 'object: got \\[1, 2, 3\\]'):
            loader.load_object([1, 2, 3])

        # test load object
        loader.load_object({
            'nested1': Config(a=1230),
            'nested1.b': 456,
            'nested2.c': '7890',
            'nested2': {'d': 'hello'}
        })
        cls_name = 'ConfigLoaderTestCase.test_load_object.<locals>.MyConfig'
        self.assertEqual(
            repr(loader.get()),
            f'{cls_name}('
            f'nested1={cls_name}.nested1(a=1230, b=456.0), '
            f'nested2={cls_name}.nested2(c=7890, d=\'hello\'))'
        )

        # test load object error
        with pytest.raises(ValueError,
                           match='at .nested1.a: cannot merge an object '
                                 'attribute into a non-object attribute'):
            loader.load_object({'nested1.a': 123,
                                'nested1': {'a': Config(value=456)}})

    def test_load_json(self):
        with TemporaryDirectory() as temp_dir:
            json_file = os.path.join(temp_dir, 'test.json')
            with codecs.open(json_file, 'wb', 'utf-8') as f:
                f.write('{"a": 1, "nested.b": 2}\n')

            loader = ConfigLoader(Config)
            loader.load_json(json_file)
            self.assertEqual(
                repr(loader.get()), 'Config(a=1, nested=Config(b=2))')

    def test_load_yaml(self):
        with TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, 'test.yaml')
            with codecs.open(yaml_file, 'wb', 'utf-8') as f:
                f.write('a: 1\nnested.b: 2\n')

            loader = ConfigLoader(Config)
            loader.load_yaml(yaml_file)
            self.assertEqual(
                repr(loader.get()), 'Config(a=1, nested=Config(b=2))')


class ValidatorTestCase(unittest.TestCase):

    def test_ValidationContext(self):
        context = ValidationContext()
        context.get_path()
        with context.enter('.a'):
            assert (context.get_path() == '.a')
            with context.enter('.b'):
                assert (context.get_path() == '.a.b')
            assert (context.get_path() == '.a')
        assert (context.get_path() == '')

    def test_IntValidator(self):
        v = IntValidator()

        self.assertEqual(v.validate(123), 123)
        self.assertEqual(v.validate(123.), 123)
        self.assertEqual(v.validate('123'), 123)

        with pytest.raises(ConfigValidationError,
                           match='casting a float number into integer is not '
                                 'allowed'):
            _ = v.validate(123.5)

        with pytest.raises(ConfigValidationError,
                           match='invalid literal for int'):
            _ = v.validate('xxx')

        with pytest.raises(ConfigValidationError,
                           match='value is not an integer'):
            _ = v.validate(123., ValidationContext(strict=True))

    def test_FloatValidator(self):
        v = FloatValidator()

        self.assertEqual(v.validate(123), 123.)
        self.assertEqual(v.validate(123.), 123.)
        self.assertEqual(v.validate(123.5), 123.5)
        self.assertEqual(v.validate('123.5'), 123.5)

        with pytest.raises(ConfigValidationError,
                           match='could not convert string to float'):
            _ = v.validate('xxx')

        with pytest.raises(ConfigValidationError,
                           match='value is not a float number'):
            _ = v.validate(123, ValidationContext(strict=True))

    def test_BoolValidator(self):
        v = BoolValidator()

        self.assertEqual(v.validate(True), True)
        self.assertEqual(v.validate('TRUE'), True)
        self.assertEqual(v.validate('On'), True)
        self.assertEqual(v.validate('yes'), True)
        self.assertEqual(v.validate(1), True)

        self.assertEqual(v.validate(False), False)
        self.assertEqual(v.validate('false'), False)
        self.assertEqual(v.validate('OFF'), False)
        self.assertEqual(v.validate('No'), False)
        self.assertEqual(v.validate(0), False)

        with pytest.raises(ConfigValidationError,
                           match='value cannot be casted into boolean'):
            _ = v.validate('xxx')

        with pytest.raises(ConfigValidationError,
                           match='value is not a boolean'):
            _ = v.validate(1, ValidationContext(strict=True))

    def test_StrValidator(self):
        v = StrValidator()

        self.assertEqual(v.validate(''), '')
        self.assertEqual(v.validate('text'), 'text')
        self.assertEqual(v.validate(123), '123')
        self.assertEqual(v.validate(True), 'True')
        self.assertEqual(v.validate(None), 'None')

        with pytest.raises(ConfigValidationError,
                           match='value is not a string'):
            _ = v.validate(1, ValidationContext(strict=True))

    def test_ConfigValidator(self):
        # check construction error
        with pytest.raises(TypeError,
                           match='`config_cls` is not Config class or a '
                                 'sub-class of Config: <class \'str\'>'):
            _ = ConfigValidator(str)

        # check validation error
        class MyConfig(Config):
            a = 123

        validator = ConfigValidator(MyConfig)
        with pytest.raises(ConfigValidationError,
                           match='value is not a ValidatorTestCase.'
                                 'test_ConfigValidator.<locals>.MyConfig'):
            validator.validate(Config(), ValidationContext(strict=True))

        # check validate_all = True
        context = ValidationContext(validate_all=True)
        self.assertEqual(validator.validate('hello', context),
                         'hello')
        with pytest.raises(ConfigValidationError,
                           match='value cannot be casted into '
                                 'ValidatorTestCase.test_ConfigValidator.'
                                 '<locals>.MyConfig'):
            context.throw()

        context = ValidationContext(validate_all=True, strict=True)
        c = Config()
        self.assertIs(validator.validate(c, context), c)
        with pytest.raises(ConfigValidationError,
                           match='value is not a ValidatorTestCase.'
                                 'test_ConfigValidator.<locals>.MyConfig'):
            context.throw()

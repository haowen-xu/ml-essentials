import codecs
import copy
import inspect
import json
import os
import re
from argparse import Action, ArgumentParser
from collections import OrderedDict, namedtuple
from contextlib import contextmanager
from os import PathLike
from typing import *

import numpy as np
import yaml
from terminaltables import AsciiTable

from .utils import NOT_SET

__all__ = [
    'ConfigAttributeNotSetError', 'ConfigValidationError',
    'ConfigField', 'Config', 'ConfigLoader', 'format_key_values',
]

ValidatorFunctionType = Callable[[Any], Any]


class ConfigAttributeNotSetError(ValueError):
    """Indicate the value of a config attribute has not been set."""

    def __init__(self, name: str):
        super().__init__(name)

    @property
    def name(self) -> str:
        """Get the name of the attribute."""
        return self.args[0]

    def __str__(self):
        return f'config attribute not set: {self.name}'


ValidationErrorInfo = namedtuple('ValidationErrorInfo', ['path', 'message'])


class ConfigValidationError(ValueError):
    """Indicate a validation error has occurred."""

    def __init__(self, errors: List[ValidationErrorInfo]):
        super().__init__(errors)

    @property
    def errors(self) -> List[ValidationErrorInfo]:
        return self.args[0]

    def __str__(self):
        get_position = lambda e: f'at {e.path}: ' if e.path else ''
        return '\n'.join(f'{get_position(e)}{e.message}' for e in self.errors)


class ConfigField(object):
    """
    Represent a config attribute field.
    """

    def __init__(self,
                 type: Optional[Type] = None,
                 default: Any = NOT_SET,
                 description: Optional[str] = None,
                 nullable: bool = True,
                 choices: Optional[Iterable] = None,
                 envvar: Optional[str] = None,
                 ignore_empty_env: bool = True,
                 validator_fn: Optional[ValidatorFunctionType] = None):
        """
        Construct a new :class:`ConfigField`.

        Args:
            type: The value type.
            default: The default config value.
                :obj:`None` if not specified.
            description: The config description.
            nullable: Whether or not :obj:`None` is a valid value?
            choices: Optional valid choices for the config value.
            envvar: Key of the environment variable, used as the default
                value for this config field.
            ignore_empty_env: Whether or not to ignore empty values from
                the environmental variable?
            validator_fn: Custom validator function for input values.
        """
        nullable = bool(nullable)

        self._type = type
        self._default_value = default
        self._description = description
        self._nullable = nullable
        self._choices = tuple(choices) if choices is not None else None
        self._envvar = envvar
        self._ignore_empty_env = ignore_empty_env
        self._validator_fn = validator_fn

    def set_type(self, type_) -> 'ConfigField':
        """Construct a new :class:`ConfigField` with new `type`."""
        return ConfigField(
            type=type_,
            default=self.default_value,
            description=self.description,
            nullable=self.nullable,
            choices=self.choices,
            envvar=self.envvar,
            ignore_empty_env=self.ignore_empty_env,
            validator_fn=self.validator_fn
        )

    def __repr__(self):
        attributes = []
        if self.type is not None:
            attributes.append(f'type={self.type.__qualname__}')
        if self.default_value is not NOT_SET:
            attributes.append(f'default={self.default_value!r}')
        attributes.append(f'nullable={self.nullable}')
        if self.choices:
            attributes.append(f'choices={list(self.choices)}')
        if self.envvar:
            attributes.append(f'envvar={self.envvar!r}')
        if self.validator_fn is not None:
            attributes.append(f'custom validator')

        return f'ConfigField({", ".join(attributes)})'

    @property
    def type(self) -> Optional[Type]:
        """Get the value type."""
        return self._type

    @property
    def default_value(self):
        """Get the default config value."""
        return self._default_value

    def get_default_value(self):
        """
        Get a copy of the default config value.

        The default value will be copied instead of directly returned.

        >>> list_value = [1, 2, 3]
        >>> field = ConfigField(list, default=list_value)
        >>> field.get_default_value() == list_value
        True
        >>> field.get_default_value() is list_value
        False

        If ``self.envvar`` is not None and ``os.environ[self.envvar]`` exists,
        the value from the environment variable will be used instead of
        the value of ``self.default_value``.
        """
        # try to get the value from env var
        val = None

        if self.envvar is not None and self.envvar in os.environ:
            val = os.environ.get(self.envvar)
            if not val and self.ignore_empty_env:
                val = None

        if val is not None:
            return get_validator(self).validate(os.environ[self.envvar])

        # otherwise return the default value configured in code
        return deep_copy(self._default_value)

    @property
    def description(self) -> Optional[str]:
        """Get the config description."""
        return self._description

    @property
    def nullable(self) -> bool:
        """Whether or not :obj:`None` is a valid value?"""
        return self._nullable

    @property
    def choices(self) -> Optional[tuple]:
        """Get the valid choices of the config value."""
        return self._choices

    @property
    def envvar(self) -> Optional[str]:
        """Get the key of the environment variable."""
        return self._envvar

    @property
    def ignore_empty_env(self) -> bool:
        """
        Whether or not to ignore empty values from the environmental variable?
        """
        return self._ignore_empty_env

    @property
    def validator_fn(self) -> Optional[ValidatorFunctionType]:
        """Get the custom validator function."""
        return self._validator_fn


class Config(object):
    """
    Base class for config objects.

    Inherit from :class:`Config`, and define config values as public class
    attributes:

    >>> class YourConfig(Config):
    ...     max_epoch = 100
    ...     max_step = ConfigField(int)
    ...     activation = ConfigField(
    ...         str, default='leaky_relu',
    ...         choices=['sigmoid', 'relu', 'leaky_relu'])
    ...     l2_regularization: float = None
    ...     train = Config(batch_size=64)  # nested config object
    ...
    ...     class test(Config):  # nested config class
    ...         batch_size = 256

    Then you may construct an instance of :class:`YourConfig` to access
    these config attributes:

    >>> config = YourConfig()
    >>> config.max_epoch
    100
    >>> config.activation
    'leaky_relu'
    >>> config.l2_regularization is None
    True
    >>> config.train.batch_size
    64
    >>> config.test.batch_size
    256

    Nested :class:`Config` objects will be copied:

    >>> config.train is YourConfig.train
    False

    Nested :class:`Config` sub-classes will be converted into instances:

    >>> YourConfig.test
    <class 'mltk.config.YourConfig.test'>
    >>> config.test
    YourConfig.test(batch_size=256)

    A :class:`ConfigAttributeNotSetError` will be raised if an attributes
    is accessed, which is defined by :class:`ConfigField` without a default
    value, and no user value has been set:

    >>> config.max_step
    Traceback (most recent call last):
        ...
    mltk.config.ConfigAttributeNotSetError: ...

    You may add a new config attribute by simply set its value:

    >>> config.new_attribute = Config(value=123)
    >>> config.new_attribute.value
    123

    You may access the config attributes via a dict-like interface:

    >>> config = YourConfig()
    >>> 'max_epoch' in config
    True
    >>> 'max_step' in config
    False
    >>> config['activation']
    'leaky_relu'
    >>> config['l2_regularization'] = 0.001
    >>> config.l2_regularization
    0.001
    >>> sorted(config)
    ['activation', 'l2_regularization', 'max_epoch', 'test', 'train']
    >>> config['not_exist']
    Traceback (most recent call last):
        ...
    KeyError: 'not_exist'

    Notes:

        Preserved names cannot be set as config attribute.  For example:

        >>> config['to_dict'] = 123
        Traceback (most recent call last):
            ...
        KeyError: 'to_dict'

        The values assigned to the attributes of a config object will not be
        validated until :meth:`validate()` is called.  For example:

        >>> config = YourConfig(max_step=1000)
        >>> config.activation = 'invalid'
        >>> config.activation
        'invalid'
        >>> config.validate()
        Traceback (most recent call last):
            ...
        mltk.config.ConfigValidationError: at .activation: value is not one of: ['sigmoid', 'relu', 'leaky_relu']

        See :meth:`validate()` for more details about validation.
    """

    def __init__(self, **kwargs):
        for key in dir(self.__class__):
            if is_config_attribute(self, key):
                def_val = getattr(self.__class__, key)

                if isinstance(def_val, type) and issubclass(def_val, Config):
                    val = def_val()
                elif isinstance(def_val, ConfigField):
                    val = def_val.get_default_value()
                elif isinstance(def_val, Config):
                    val = def_val.copy()
                else:
                    val = deep_copy(def_val)

                setattr(self, key, val)

        self.update(kwargs)

    def __repr__(self):
        attributes = [
            f'{key}={getattr(self, key)!r}' for key in sorted(self)
        ]
        return f'{self.__class__.__qualname__}({", ".join(attributes)})'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        keys = sorted(self)
        if keys != sorted(other):
            return False
        for key in keys:
            if self[key] != other[key]:
                return False
        return True

    def __hash__(self):
        return hash(tuple([(key, self[key]) for key in sorted(self)]))

    def __getattribute__(self, name):
        val = object.__getattribute__(self, name)
        if val is NOT_SET:
            raise ConfigAttributeNotSetError(name)
        return val

    def __setattr__(self, name, value):
        if '.' in name:
            raise AttributeError(f'`name` must not contain \'.\': {name!r}')
        if isinstance(value, ConfigField):
            raise TypeError(f'`value` must not be a ConfigField: got {value!r}')
        object.__setattr__(self, name, value)

    def __contains__(self, key):
        try:
            return (
                key in self.__dict__ and
                getattr(self, key) is not NOT_SET and
                is_config_attribute(self, key)
            )
        except ConfigAttributeNotSetError:
            return False

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        if not is_config_attribute(self, key, require_existence=False):
            raise KeyError(key)
        setattr(self, key, value)

    def __iter__(self):
        return (key for key in self.__dict__ if key in self)

    def copy(self, **kwargs):
        """
        Get a copy of a config object:

        >>> class YourConfig(Config):
        ...     nested = Config(c=789)

        >>> config = YourConfig(a=123, b=456)
        >>> copied = config.copy(a=1230)
        >>> isinstance(copied, YourConfig)
        True
        >>> copied.a
        1230
        >>> copied.b
        456
        >>> copied.nested.c
        789
        >>> copied.nested is config.nested
        False

        Args:
            \\**kwargs: The new key and values to be set.

        Returns:
            The copied config object.
        """
        ret = deep_copy(self)
        for key, val in kwargs.items():
            setattr(ret, key, val)
        return ret

    def to_dict(self, deepcopy=False) -> dict:
        """
        Convert a config object to a dict.

        All the nested :class:`Config` instances will be converted into dict.

        >>> config = Config(a=123, nested=Config(b=456))
        >>> config.to_dict()
        {'a': 123, 'nested': {'b': 456}}

        For object attributes, they will not be copied by default, unless
        `deepcopy = True`:

        >>> config = Config(value=[1, 2, 3])
        >>> config.value
        [1, 2, 3]
        >>> config.to_dict()['value'] is config.value
        True
        >>> config.to_dict(deepcopy=True)['value'] is config.value
        False

        Args:
            deepcopy: Whether or not to deep copy the attribute values?

        Returns:
            The config dict.
        """
        ret = {}
        for key in self:
            val = self[key]
            if isinstance(val, Config):
                ret[key] = val.to_dict()
            else:
                if deepcopy:
                    val = deep_copy(val)
                ret[key] = val
        return ret

    def to_flatten_dict(self, deepcopy=False) -> dict:
        """
        Convert a config object to a flatten dict.

        All values from the nested :class:`Config` and :class:`dict` instances
        will be gathered into the returned dict.

        >>> config = Config(a=123, nested=Config(b=456))
        >>> config.to_flatten_dict()
        {'a': 123, 'nested.b': 456}

        For object attributes, they will not be copied by default, unless
        `deepcopy = True`:

        >>> config = Config(value={'a': [1, 2, 3]})
        >>> config2 = config.to_flatten_dict()
        >>> config2
        {'value.a': [1, 2, 3]}
        >>> config2['value.a'] is config.value['a']
        True
        >>> config2 = config.to_flatten_dict(deepcopy=True)
        >>> config2['value.a'] is config.value['a']
        False

        Args:
            deepcopy: Whether or not to deep copy the attribute values?

        Returns:
            The flatten config dict.

        Notes:
            The flatten dict can be converted back to a :class:`Config`
            instance by :meth:`ConfigLoader.load_object()`.
        """
        def flatten(c, prefix):
            for key in c:
                val = c[key]
                if isinstance(val, (Config, dict)):
                    flatten(val, f'{prefix}{key}.')
                else:
                    if deepcopy:
                        ret[f'{prefix}{key}'] = deep_copy(val)
                    else:
                        ret[f'{prefix}{key}'] = val
        ret = {}
        flatten(self, '')

        return ret

    @classmethod
    def defaults_dict(cls) -> dict:
        """
        Get the default values as a dict:

        >>> class YourConfig(Config):
        ...     a = 123
        ...     b = ConfigField(int, default=456)
        ...     missing: int
        ...     missing2 = ConfigField(int)
        ...
        ...     class nested(Config):
        ...         c = 789

        >>> YourConfig.defaults_dict()
        {'a': 123, 'b': 456, 'nested': {'c': 789}}

        Returns:
            The default config dict.
        """
        return cls().to_dict()

    @classmethod
    def defaults_flatten_dict(cls) -> dict:
        """
        Get the default values as a flatten dict:

        >>> class YourConfig(Config):
        ...     a = 123
        ...     b = ConfigField(int, default=456)
        ...     missing: int
        ...     missing2 = ConfigField(int)
        ...
        ...     class nested(Config):
        ...         c = 789

        >>> YourConfig.defaults_flatten_dict()
        {'a': 123, 'b': 456, 'nested.c': 789}

        Returns:
            The default config flatten dict.
        """
        return cls().to_flatten_dict()

    def update(self, key_values: Union['Config', dict]):
        """
        Recursively update all nested config objects.

        Directly setting the config attributes will override any nested
        config object using the specified new value.  For example:

        >>> config = Config(nested=Config(value=123))
        >>> config.nested
        Config(value=123)
        >>> config.nested = {'new_value': 123}
        >>> config.nested
        {'new_value': 123}

        If you intend to update the nested config objects using the new
        values, you should use :meth:`update()` as follows:

        >>> config = Config(nested=Config(value=123))
        >>> config.update({'nested': {'value2': 456}})
        >>> config.nested
        Config(value=123, value2=456)

        It is also possible to call :meth:`update()` with another config object:

        >>> config = Config(nested=Config(value=123))
        >>> config.update(Config(nested=Config(value=1230, value2=456)))
        >>> config.nested
        Config(value=1230, value2=456)

        Args:
            key_values: The new key and values.

        Notes:
            When loading config attributes from multiple dicts by
            :meth:`update()`, it is important to call :meth:`validate()`
            after each call to :meth:`update()`.
            This will correctly convert all nested dicts into nested config
            objects, for example:

            >>> class ParentConfig(Config):
            ...     child = ConfigField(Config)

            >>> config = ParentConfig()
            >>> config.update({'child': {'a': 1}})
            >>> config.validate()
            ParentConfig(child=Config(a=1))
            >>> config.update({'child': {'b': 2}})
            >>> config.validate()
            ParentConfig(child=Config(a=1, b=2))

            In contrary, if :meth:`validate()` is not called, then the second
            update will override the first update:

            >>> config = ParentConfig()
            >>> config.update({'child': {'a': 1}})
            >>> config.update({'child': {'b': 2}})
            >>> config.validate()
            ParentConfig(child=Config(b=2))

            Sometimes the first few updates will not set all missing fields,
            in which cases :meth:`validate(ignore_missing=True)` should be used.
            For example:

            >>> config = ParentConfig()
            >>> config.update({'new_value': 123})
            >>> config.validate()  # will fail with an error
            Traceback (most recent call last):
                ...
            mltk.config.ConfigValidationError: at .child: config attribute is required but not set
            >>> config.validate(ignore_missing=True)  # okay
            ParentConfig(new_value=123)
        """
        for key in key_values:
            val = key_values[key]

            try:
                self_val = getattr(self, key, None)
            except ConfigAttributeNotSetError:
                self_val = None

            if isinstance(self_val, Config) and \
                    isinstance(val, (Config, dict, OrderedDict)):
                self_val.update(val)
            else:
                setattr(self, key, val)

    def user_validate(self) -> None:
        """
        Validate the attributes and convert the values into desired types,
        designed to be implemented by user.

        >>> class YourConfig(Config):
        ...     value = 123
        ...
        ...     def user_validate(self):
        ...         if self.value < 100:
        ...             raise ValueError('`value` must >= 100.')

        >>> config = YourConfig()
        >>> config.validate()
        YourConfig(value=123)
        >>> config.value = 99
        >>> config.validate()
        Traceback (most recent call last):
            ...
        mltk.config.ConfigValidationError: `value` must >= 100.
        """

    def validate(self, ignore_missing: bool = False,
                 validate_all: bool = False):
        """
        Validate the attributes, and convert the values into desired types.

        :meth:`validate()` will not only raise error if any attribute cannot
        pass validation.  It will also convert the attribute values into their
        desired types.  Note the return value of :meth:`validate()` is the
        config object itself.  For example:

        >>> class YourConfig(Config):
        ...     a = 123  # int attribute
        ...     b = ConfigField(float)  # float attribute, without default value
        ...     c: float  # float attribute, without default value
        ...
        ...     class nested(Config):
        ...         d = ConfigField(default=10000)

        >>> config = YourConfig(a='1230', b='0.001', c='1.5')
        >>> config
        YourConfig(a='1230', b='0.001', c='1.5', nested=YourConfig.nested(d=10000))
        >>> validated = config.validate()
        >>> validated
        YourConfig(a=1230, b=0.001, c=1.5, nested=YourConfig.nested(d=10000))
        >>> validated is config
        True

        If the expected type of a config attribute is :class:`Config` or any
        sub-class of :class:`Config`, and the value assigned to it is a dict
        or a config object, its type will be converted to the correct config
        type, if necessary.  For example:

        >>> config = YourConfig(b=456, c=789)
        >>> config.nested = {'d': '10000'}
        >>> config
        YourConfig(a=123, b=456, c=789, nested={'d': '10000'})
        >>> config.validate()
        YourConfig(a=123, b=456.0, c=789.0, nested=YourConfig.nested(d=10000))

        >>> config = YourConfig(b=456, c=789)
        >>> config.nested = Config()
        >>> config
        YourConfig(a=123, b=456, c=789, nested=Config())
        >>> config.validate()
        YourConfig(a=123, b=456.0, c=789.0, nested=YourConfig.nested(d=10000))

        >>> class NestedSubclass(YourConfig.nested):
        ...     e = 'hello'

        >>> config = YourConfig(b=456, c=789)
        >>> config.nested = NestedSubclass()
        >>> config
        YourConfig(a=123, b=456, c=789, nested=NestedSubclass(d=10000, e='hello'))
        >>> config.validate()
        YourConfig(a=123, b=456.0, c=789.0, nested=NestedSubclass(d=10000, e='hello'))

        If you want to collect all the errors, rather than the first error,
        you may pass `validate_all = True`.  For example:

        >>> config = YourConfig(a='invalid int', b='invalid float', c='invalid float2')
        >>> config.validate(validate_all=True)
        Traceback (most recent call last):
            ...
        mltk.config.ConfigValidationError: at .a: invalid literal for int() with base 10: 'invalid int'
        at .b: could not convert string to float: 'invalid float'
        at .c: could not convert string to float: 'invalid float2'

        Args:
            ignore_missing: Whether or not to ignore missing attribute?
                (i.e., attribute defined by :class:`ConfigField` without
                a default value, to which the user has not set a value)
            validate_all: Whether or not to validate all fields even if
                some attribute already has an error?

        Returns:
            The validated config object.
        """
        context = ValidationContext(ignore_missing=ignore_missing,
                                    validate_all=validate_all)
        ret = ConfigValidator(self.__class__).validate(self, context)
        if validate_all:
            context.throw()
        assert(isinstance(ret, self.__class__))
        return ret


TConfig = TypeVar('TConfig')


class ConfigLoader(Generic[TConfig]):
    """
    A class to help load config attributes from multiple sources.
    """

    def __init__(self, config_or_cls: Union[Type[TConfig], TConfig],
                 validate_all: bool = False):
        """
        Construct a new :class:`ConfigLoader`.

        Args:
            config_or_cls: A config object, or a config class.
            validate_all: Whether or not to validate all fields even if
                some attribute already has an error?
        """
        if isinstance(config_or_cls, type):
            config_cls = config_or_cls
            config = config_or_cls()
        else:
            config_cls = config_or_cls.__class__
            config = config_or_cls

        if not issubclass(config_cls, Config):
            raise TypeError(f'`config_or_cls` is neither a Config class, '
                            f'nor a Config instance: {config_or_cls!r}')

        self._config_cls = config_cls
        self._config = config
        self._validate_all = validate_all

    @property
    def config_cls(self) -> Type[TConfig]:
        return self._config_cls

    @property
    def validate_all(self) -> bool:
        return self._validate_all

    def get(self, ignore_missing: bool = False) -> TConfig:
        """
        Get the validated config object.

        Args:
            ignore_missing: Whether or not to ignore missing attribute?
                (i.e., attribute defined by :class:`ConfigField` without
                a default value, to which the user has not set a value)
        """
        return self._config.validate(ignore_missing=ignore_missing,
                                     validate_all=self.validate_all)

    def load_object(self, key_values: Union[dict, Config]):
        """
        Load config attributes from the specified `key_values` object.

        All nested dicts will be converted into config objects.
        Also, all "." in keys will be further parsed into nested objects.
        For example:

        >>> class ConfigNested1(Config):
        ...     a = 123
        ...     b = ConfigField(float, default=None)

        >>> class YourConfig(Config):
        ...     nested1: ConfigNested1
        ...
        ...     class nested2(Config):
        ...         c = 789

        >>> loader = ConfigLoader(YourConfig)
        >>> loader.load_object({'nested1': Config(a=1230)})
        >>> loader.load_object({'nested2.c': '7890'})
        >>> loader.load_object(Config(nested1=Config(b=456)))
        >>> loader.load_object({'nested2.d': {'even_nested.value': 'hello'}})
        >>> loader.get()
        YourConfig(nested1=ConfigNested1(a=1230, b=456.0), nested2=YourConfig.nested2(c=7890, d=Config(even_nested=Config(value='hello'))))

        If the full name of some non-object config attribute collides with
        some object attribute in one :meth:`load_object()` call, then an
        error will be raised, for example:

        >>> loader = ConfigLoader(Config)
        >>> loader.load_object({'nested1.a': 1230, 'nested1': 'literal'})
        Traceback (most recent call last):
            ...
        ValueError: at .nested1: cannot merge a non-object attribute into an object attribute
        >>> loader.load_object({'nested1': 'literal', 'nested1.a': 1230})
        Traceback (most recent call last):
            ...
        ValueError: at .nested1.a: cannot merge an object attribute into a non-object attribute

        Args:
            key_values: The dict or config object.
        """
        if not isinstance(key_values, (dict, Config)):
            raise TypeError(f'`key_values` must be a dict or a Config object: '
                            f'got {key_values!r}')

        def copy_values(src, dst, prefix):
            for key in src:
                err_msg1 = lambda: (
                    f'at {prefix + key}: cannot merge a non-object '
                    f'attribute into an object attribute')
                err_msg2 = lambda: (
                    f'at {prefix + key}: cannot merge an object '
                    f'attribute into a non-object attribute')

                # find the target node in dst
                parts = key.split('.')
                tmp = dst
                for part in parts[:-1]:
                    if part not in tmp:
                        tmp[part] = Config()
                    elif not isinstance(tmp[part], Config):
                        raise ValueError(err_msg2())
                    tmp = tmp[part]

                # get the src and dst values
                part = parts[-1]
                src_val = src[key]
                try:
                    dst_val = getattr(tmp, part)
                except (ConfigAttributeNotSetError, AttributeError):
                    dst_val = NOT_SET

                # now copy the values to the target node
                if isinstance(src_val, (dict, Config)):
                    if dst_val is NOT_SET:
                        new_val = copy_values(
                            src_val, Config(), prefix=prefix + key + '.')
                    elif isinstance(dst_val, Config):
                        new_val = copy_values(
                            src_val, dst_val, prefix=prefix + key + '.')
                    else:
                        raise ValueError(err_msg2())
                elif isinstance(src_val, StrictDict):
                    new_val = src_val.value
                else:
                    if isinstance(dst_val, Config):
                        raise ValueError(err_msg1())
                    else:
                        new_val = src_val

                tmp[part] = new_val

            return dst

        self._config.update(copy_values(key_values, Config(), prefix='.'))

    def load_json(self, path: Union[str, bytes, PathLike], cls=None) -> TConfig:
        """
        Load config from a JSON file.

        Args:
            path: Path of the JSON file.
            cls: The JSON decoder class.
        """
        with codecs.open(path, 'rb', 'utf-8') as f:
            obj = json.load(f, cls=cls)
            self.load_object(obj)

    def load_yaml(self, path: Union[str, bytes, PathLike],
                  Loader=yaml.SafeLoader) -> TConfig:
        """
        Load config from a YAML file.

        Args:
            path: Path of the YAML file.
            Loader: The YAML loader class.
        """
        with codecs.open(path, 'rb', 'utf-8') as f:
            obj = yaml.load(f, Loader=Loader)
            if obj is not None:
                self.load_object(obj)

    def load_file(self, path: Union[str, bytes, PathLike]) -> TConfig:
        """
        Load config from a file.

        The file will be loaded according to its extension.  Supported
        extensions are::

            *.yml, *.yaml, *.json

        Args:
            path: Path of the file.
        """
        name, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext in ('.yml', '.yaml'):
            self.load_yaml(path)
        elif ext in ('.json',):
            self.load_json(path)
        else:
            raise IOError(f'Unsupported config file extension: {ext}')

    def build_arg_parser(self, parser: Optional[ArgumentParser] = None) \
            -> ArgumentParser:
        """
        Build an argument parser.

        This method is a sub-procedure of :class:`parse_args()`.
        Un-specified options will be :obj:`NOT_SET` in the namespace
        returned by the parser.

        Args:
            parser: The parser to populate the arguments.
                If not specified, will create a new parser.

        Returns:
            The argument parser.
        """

        class _ConfigAction(Action):

            def __init__(self, validator: Validator, option_strings, dest,
                         **kwargs):
                super().__init__(option_strings, dest, **kwargs)
                self._validator = validator

            def __call__(self, parser, namespace, values,
                         option_string=None):
                try:
                    if self._validator is None or (
                            isinstance(self._validator, FieldValidator) and
                            self._validator.sub_validator is None):
                        try:
                            value = yaml.load(values, Loader=yaml.SafeLoader)
                        except yaml.YAMLError:
                            value = str(values)
                    else:
                        context = ValidationContext()
                        with context.enter(f'.{self.dest}'):
                            value = self._validator.validate(values,
                                                             context)

                except Exception as ex:
                    message = f'Invalid value for argument `{option_string}`'
                    if str(ex):
                        message += '; ' + str(ex)
                    if not message.endswith('.'):
                        message += '.'
                    raise ValueError(message)
                else:
                    if isinstance(value, dict):
                        value = StrictDict(value)
                    setattr(namespace, self.dest, value)

        # gather the nested config fields
        def get_field_help(field):
            config_help = field.description or ''
            default_value = field.get_default_value()
            if config_help:
                config_help += ' '
            if default_value is NOT_SET:
                config_help += '(required'
            else:
                config_help += f'(default {default_value!r}'
            if field.choices:
                config_help += f'; choices {sorted(field.choices)}'
            config_help += ')'
            return config_help

        def gather_args(cls, prefix):
            if prefix:
                prefix += '.'

            annotations = getattr(cls, '__annotations__', {})
            annotated_keys = list(annotations)

            for key in sorted(set(list(dir(cls)) + annotated_keys)):
                if is_config_attribute(cls, key):
                    val = getattr(cls, key, None)
                    validator = get_validator(val, annotations.get(key))

                    if isinstance(validator, ConfigValidator):
                        gather_args(validator.config_cls, prefix + key)
                    elif isinstance(validator, FieldValidator) and \
                            isinstance(validator.sub_validator,
                                       ConfigValidator):
                        gather_args(validator.sub_validator.config_cls,
                                    prefix + key)
                    else:
                        option_string = f'--{prefix}{key}'

                        if isinstance(val, ConfigField):
                            help_msg = get_field_help(val)
                        else:
                            help_msg = f'(default {val})'

                        parser.add_argument(
                            option_string, help=help_msg,
                            action=_ConfigAction, validator=validator,
                            default=NOT_SET
                        )

        # populate the arguments
        if parser is None:
            parser = ArgumentParser()
        gather_args(self.config_cls, '')

        return parser

    def parse_args(self, args: Iterable[str]):
        """
        Parse config attributes from CLI argument.

        >>> class YourConfig(Config):
        ...     a = 123
        ...     b = ConfigField(float, description="a float number")
        ...
        ...     class nested(Config):
        ...         c = ConfigField(str, choices=['hello', 'bye'])

        >>> loader = ConfigLoader(YourConfig)
        >>> loader.parse_args([
        ...     '--a=1230',
        ...     '--b=456',
        ...     '--nested.c=hello'
        ... ])
        >>> loader.get()
        YourConfig(a=1230, b=456.0, nested=YourConfig.nested(c='hello'))

        Args:
            args: The CLI arguments.
        """
        parser = self.build_arg_parser()
        namespace = parser.parse_args(list(args))
        parsed = {key: value for key, value in vars(namespace).items()
                  if value is not NOT_SET}
        self.load_object(parsed)


TValue = TypeVar('TValue')
PatternType = type(re.compile('x'))


def deep_copy(value: TValue) -> TValue:
    """
    A patched deep copy function, that can handle various types cannot be
    handled by the standard :func:`copy.deepcopy`.

    Args:
        value: The value to be copied.

    Returns:
        The copied value.
    """
    def pattern_dispatcher(v, memo=None):
        return v  # we don't need to copy a regex pattern object, it's read-only

    old_dispatcher = copy._deepcopy_dispatch.get(PatternType, None)
    copy._deepcopy_dispatch[PatternType] = pattern_dispatcher
    try:
        return copy.deepcopy(value)
    finally:
        if old_dispatcher is not None:  # pragma: no cover
            copy._deepcopy_dispatch[PatternType] = old_dispatcher
        else:
            del copy._deepcopy_dispatch[PatternType]


class StrictDict(object):
    """
    Class to wrap a dict, such that :meth:`ConfigLoader.load_object()`
    will not interpret it as a nested config object.
    """

    def __init__(self, value):
        self.value = value


def is_config_attribute(cls_or_obj: Union[Type['Config'], 'Config'],
                        name: str, require_existence: bool = True) -> bool:
    """
    Test whether or not `name` is a config attribute of `cls_or_obj`.

    An attribute is a config attribute if it is a public data attribute
    (i.e., not a method, not a property, not starting with "_"), and is
    not a direct member of the base :class:`Config` class.

    Args:
        cls_or_obj: A subclass of :class:`Config`, or an instance of
            any subclass of :class:`Config`.
        name: The attribute name to be tested.
        require_existence: Whether or not to require the existence of the
            attribute?

    Returns:
        Whether or not `name` is a config attribute.
    """
    if name.startswith('_') or hasattr(Config, name):
        return False

    annotated = (isinstance(cls_or_obj, type) and
                 hasattr(cls_or_obj, '__annotations__') and
                 name in cls_or_obj.__annotations__)

    try:
        if not annotated and not hasattr(cls_or_obj, name):
            return not require_existence
    except ConfigAttributeNotSetError:
        return True

    if isinstance(cls_or_obj, type):
        cls_val = getattr(cls_or_obj, name, None)
    else:
        cls_val = getattr(cls_or_obj.__class__, name, None)

    if not isinstance(cls_val, type):
        if isinstance(cls_val, property) or \
                inspect.ismethod(cls_val) or \
                inspect.isfunction(cls_val):
            return False

    return True


class ValidationContext(object):
    """Maintain the context for validation."""

    def __init__(self, strict: bool = False, ignore_missing: bool = False,
                 validate_all: bool = False):
        """
        Construct a new :class:`ValidatorContext`.

        Args:
            strict: If :obj:`True`, disable type conversion.
                If :obj:`False`, the validator will try its best to convert the
                input `value` into desired type.
            ignore_missing: Whether or not to ignore missing attribute?
                (i.e., attribute defined by :class:`ConfigField` without
                a default value, to which the user has not set a value)
            validate_all: Whether or not to validate all fields even if
                some attribute already has an error?
        """
        self._scopes = []
        self._errors = []
        self._strict = strict
        self._ignore_missing = ignore_missing
        self._validate_all = validate_all

    @contextmanager
    def enter(self, scope: str):
        try:
            self._scopes.append(scope)
            yield
        finally:
            self._scopes.pop()

    @property
    def strict(self) -> bool:
        return self._strict

    @property
    def ignore_missing(self) -> bool:
        return self._ignore_missing

    @property
    def validate_all(self) -> bool:
        return self._validate_all

    def get_path(self) -> str:
        return ''.join(self._scopes)

    def add_error(self, error: Union[Exception, str]):
        """
        Add a validation error.

        Args:
            error: An exception or an error message.
        """
        err = ValidationErrorInfo(self.get_path(), str(error))
        if not self.validate_all:
            raise ConfigValidationError([err])
        else:
            self._errors.append(err)

    def throw(self):
        """Throw the error."""
        if self._errors:
            raise ConfigValidationError(self._errors)


class Validator(object):
    """Base config validator."""

    def __repr__(self):
        return f'{self.__class__.__qualname__}()'

    def _validate(self, value, context: ValidationContext):
        raise NotImplementedError()

    def validate(self, value, context: Optional[ValidationContext] = None):
        """
        Validate the `value`.

        Args:
            value: The value to be validated.
            context: The context for validator.

        Returns:
            The validated value.

        Raises:
            ConfigValidationError: If the value cannot pass validation.
        """
        if context is None:
            context = ValidationContext()
        try:
            return self._validate(value, context)
        except ConfigValidationError:
            raise  # do not re-generate the validation errors
        except Exception as ex:
            context.add_error(ex)
            return value


class CustomValidator(Validator):
    """Custom validator function wrapper."""

    def __init__(self, validator_fn: ValidatorFunctionType):
        self._validator_fn = validator_fn

    @property
    def validator_fn(self) -> ValidatorFunctionType:
        """Get the validator function."""
        return self._validator_fn

    def __repr__(self):
        return f'{self.__class__.__qualname__}({self.validator_fn})'

    def _validate(self, value, context: ValidationContext):
        return self.validator_fn(value)


class FieldValidator(Validator):
    """
    Validator for a specified config attribute.

    >>> class YourConfig(Config):
    ...     a = ConfigField(int)
    ...     b = ConfigField(float, nullable=False)
    ...     c = ConfigField(str, choices=['hello', 'bye'])

    >>> validator = FieldValidator(YourConfig.a)
    >>> validator.validate('123')
    123
    >>> validator.validate(None)

    >>> validator = FieldValidator(YourConfig.b)
    >>> validator.validate('123.5')
    123.5
    >>> validator.validate(None)
    Traceback (most recent call last):
        ...
    mltk.config.ConfigValidationError: null value is not allowed

    >>> validator = FieldValidator(YourConfig.c)
    >>> validator.validate('hello')
    'hello'
    >>> validator.validate('invalid str')
    Traceback (most recent call last):
        ...
    mltk.config.ConfigValidationError: value is not one of: ['hello', 'bye']
    """

    def __init__(self, field: ConfigField):
        self._field = field
        if field.validator_fn is not None:
            self._sub_validator = CustomValidator(field.validator_fn)
        elif field.type is not None:
            self._sub_validator = get_validator(field.type)
        else:
            self._sub_validator = get_validator(field.default_value)

    def __repr__(self):
        return f'FieldValidator({self.field!r})'

    @property
    def field(self):
        return self._field

    @property
    def sub_validator(self) -> Optional[Validator]:
        return self._sub_validator

    def _validate(self, value, context):
        if value is not None and self.sub_validator is not None:
            value = self.sub_validator.validate(value, context)

        if value is None:
            if not self.field.nullable:
                context.add_error('null value is not allowed')

        else:
            if self.field.choices and value not in self.field.choices:
                context.add_error(f'value is not one of: '
                                  f'{list(self.field.choices)}')

        return value


def get_validator(type_or_value,
                  additional_type: Optional[type] = None
                  ) -> Optional[Validator]:
    """
    Get the validator for specified type or value, or config attribute.

    >>> class YourConfig(Config):
    ...     value = ConfigField(int, choices=[1, 2, 3])

    >>> get_validator(int)
    IntValidator()
    >>> get_validator(123)
    IntValidator()
    >>> get_validator(float)
    FloatValidator()
    >>> get_validator(123.5)
    FloatValidator()
    >>> get_validator(bool)
    BoolValidator()
    >>> get_validator(True)
    BoolValidator()
    >>> get_validator(str)
    StrValidator()
    >>> get_validator('hello')
    StrValidator()
    >>> get_validator(YourConfig)
    ConfigValidator(YourConfig)
    >>> get_validator(YourConfig())
    ConfigValidator(YourConfig)
    >>> get_validator(YourConfig.value)
    FieldValidator(ConfigField(type=int, nullable=True, choices=[1, 2, 3]))
    >>> get_validator(dict)
    >>> get_validator([])

    Args:
        type_or_value: The type or the value.
        additional_type: An additional type hint.

    Returns:
        The Validator, or :obj:`None` if no validator can be provided.
    """
    if isinstance(type_or_value, type):
        type_ = type_or_value
        value = None
    elif type_or_value is not None:
        type_ = type_or_value.__class__
        value = type_or_value
    else:
        type_ = additional_type
        value = type_or_value

    if type_ == int:
        return IntValidator()
    elif type_ == float:
        return FloatValidator()
    elif type_ == bool:
        return BoolValidator()
    elif type_ in (str, bytes):
        return StrValidator()
    elif isinstance(type_, type) and issubclass(type_, Config):
        return ConfigValidator(type_)
    elif isinstance(type_, type) and isinstance(value, ConfigField):
        if additional_type is not None:
            value = value.set_type(additional_type)
        return FieldValidator(value)
    else:
        return None


class IntValidator(Validator):
    """
    Validator for integer values.

    >>> validator = IntValidator()
    >>> validator.validate(123)
    123
    >>> validator.validate(123.0)
    123
    >>> validator.validate('123')
    123
    >>> validator.validate(123.5)
    Traceback (most recent call last):
        ...
    mltk.config.ConfigValidationError: ...
    """

    def _validate(self, value, context):
        if not context.strict:
            int_value = int(value)
            float_value = float(value)
            if np.abs(int_value - float_value) > np.finfo(float_value).eps:
                context.add_error(
                    'casting a float number into integer is not allowed')
            value = int_value
        if not isinstance(value, int):
            context.add_error('value is not an integer')
        return value


class FloatValidator(Validator):
    """
    Validator for float values.

    >>> validator = FloatValidator()
    >>> validator.validate(123)
    123.0
    >>> validator.validate(123.0)
    123.0
    >>> validator.validate('123.0')
    123.0
    >>> validator.validate(123.5)
    123.5
    """

    def _validate(self, value, context):
        if not context.strict:
            value = float(value)
        if not isinstance(value, float):
            context.add_error('value is not a float number')
        return value


class BoolValidator(Validator):
    """
    Validator for boolean values.

    >>> validator = BoolValidator()
    >>> validator.validate(True)
    True
    >>> validator.validate(1)
    True
    >>> validator.validate('1')
    True
    >>> validator.validate('on')
    True
    >>> validator.validate('yes')
    True
    >>> validator.validate('true')
    True
    >>> validator.validate(False)
    False
    >>> validator.validate(0)
    False
    >>> validator.validate('0')
    False
    >>> validator.validate('off')
    False
    >>> validator.validate('no')
    False
    >>> validator.validate('false')
    False
    >>> validator.validate('bad literal')
    Traceback (most recent call last):
        ...
    mltk.config.ConfigValidationError: ...
    """

    def _validate(self, value, context):
        if not context.strict:
            if isinstance(value, (str, bytes)):
                value = str(value).lower()
                if value in ('1', 'on', 'yes', 'true'):
                    value = True
                elif value in ('0', 'off', 'no', 'false'):
                    value = False
            elif isinstance(value, int):
                if value == 1:
                    value = True
                elif value == 0:
                    value = False
            if not isinstance(value, bool):
                context.add_error('value cannot be casted into boolean')
        else:
            if not isinstance(value, bool):
                context.add_error('value is not a boolean')
        return value


class StrValidator(Validator):
    """
    Validator for string values.

    >>> validator = StrValidator()
    >>> validator.validate('hello')
    'hello'
    >>> validator.validate(123)
    '123'
    """

    def _validate(self, value, context):
        if not context.strict:
            value = str(value)
        if not isinstance(value, str):
            context.add_error('value is not a string')
        return value


class ConfigValidator(Validator):
    """
    Validator for :class:`Config` objects.

    Build the validator according to a config class:

    >>> class YourConfig(Config):
    ...     max_epoch = 100
    ...     max_step: int
    ...     learning_rate = ConfigField(float, default=None)

    >>> validator = ConfigValidator(YourConfig)
    >>> value = {'max_step': 1000, 'new_value': 123}

    Dict can be casted into expected config class:

    >>> validator.validate(value)
    YourConfig(learning_rate=None, max_epoch=100, max_step=1000, new_value=123)

    Config objects which are not instances of the expected config class
    can be casted into the expected config class:

    >>> value = Config(max_epoch=200, max_step=2000)
    >>> validator.validate(value)
    YourConfig(learning_rate=None, max_epoch=200, max_step=2000)

    Config objects which are already instances of the expected config class
    are validated and returned directly:

    >>> value = YourConfig(max_step=1250)
    >>> validator.validate(value)
    YourConfig(learning_rate=None, max_epoch=100, max_step=1250)
    >>> validator.validate(value) is value
    True

    Un-set attributes will throw an error, unless `ignore_missing` is True:

    >>> value = YourConfig()
    >>> validator.validate(value)
    Traceback (most recent call last):
        ...
    mltk.config.ConfigValidationError: at .max_step: config attribute is required but not set
    >>> validator.validate(value, ValidationContext(ignore_missing=True))
    YourConfig(learning_rate=None, max_epoch=100)

    Validating incompatible objects will throw an error:

    >>> validator.validate('hello')
    Traceback (most recent call last):
        ...
    mltk.config.ConfigValidationError: value cannot be casted into YourConfig
    """

    def __init__(self, config_cls: Type[Config]):
        """
        Construct a new :class:`ConfigClassValidator`.

        Args:
            config_cls: The config class.
        """
        if not isinstance(config_cls, type) or \
                not issubclass(config_cls, Config):
            raise TypeError(f'`config_cls` is not Config class or a '
                            f'sub-class of Config: {config_cls}')
        self._config_cls = config_cls

    def __repr__(self):
        return f'{self.__class__.__qualname__}({self.config_cls.__qualname__})'

    @property
    def config_cls(self) -> Type[Config]:
        return self._config_cls

    def _validate(self, value, context):
        if not context.strict:
            if isinstance(value, self.config_cls):
                pass
            elif isinstance(value, (dict, OrderedDict, Config)):
                old_value = value
                value = self.config_cls()
                value.update(old_value)
            else:
                context.add_error(f'value cannot be casted into '
                                  f'{self.config_cls.__qualname__}')
                return value
        else:
            if not isinstance(value, self.config_cls):
                context.add_error(f'value is not a '
                                  f'{self.config_cls.__qualname__}')
                return value

        annotations = getattr(self.config_cls, '__annotations__', {})

        for key in sorted(set(list(dir(self.config_cls)) + list(annotations))):
            if is_config_attribute(self.config_cls, key):
                cls_val = getattr(self.config_cls, key, None)

                # get the validator
                validator = get_validator(cls_val, annotations.get(key))

                # validate the value
                if validator is not None:
                    with context.enter(f'.{key}'):
                        try:
                            if not hasattr(value, key):
                                # if `key` attribute is annotated rather
                                # than a ConfigField instance
                                raise ConfigAttributeNotSetError(key)
                            val = getattr(value, key)
                            if isinstance(validator, FieldValidator) or \
                                    val is not None:
                                val = validator.validate(val, context)
                        except ConfigAttributeNotSetError as ex:
                            if not context.ignore_missing:
                                context.add_error('config attribute is '
                                                  'required but not set')
                        else:
                            setattr(value, key, val)

        try:
            value.user_validate()
        except Exception as ex:
            context.add_error(ex)

        return value


KeyValuesType = Union[Dict, Config, Iterable[Tuple[str, Any]]]


def format_key_values(key_values: KeyValuesType,
                      title: Optional[str] = None,
                      formatter: Callable[[Any], str] = str,
                      delimiter_char: str = '=') -> str:
    """
    Format key value sequence into str.

    The basic usage, to format a :class:`Config`, a dict or a list of tuples:

    >>> print(format_key_values(Config(a=123, b=Config(value=456))))
    a         123
    b.value   456
    >>> print(format_key_values({'a': 123, 'b': {'value': 456}}))
    a   123
    b   {'value': 456}
    >>> print(format_key_values([('a', 123), ('b', {'value': 456})]))
    a   123
    b   {'value': 456}

    To add a title and a delimiter:

    >>> print(format_key_values(Config(a=123, b=Config(value=456)),
    ...                         title='short title'))
    short title
    =============
    a         123
    b.value   456
    >>> print(format_key_values({'a': 123, 'b': {'value': 456}},
    ...                         title='long long long title'))
    long long long title
    ====================
    a   123
    b   {'value': 456}

    Args:
        key_values: The sequence of key values, may be a :class:`Config`,
            a dict, or a list of (key, value) pairs.
            If it is a :class:`Config`, it will be flatten via
            :meth:`Config.to_flatten_dict()`.
        title: If specified, will prepend a title and a horizontal delimiter
            to the front of returned string.
        formatter: The function to format values.
        delimiter_char: The character to use for the delimiter between title
            and config key values.

    Returns:
        The formatted str.
    """
    if len(delimiter_char) != 1:
        raise ValueError(f'`delimiter_char` must be one character: '
                         f'got {delimiter_char!r}')

    if isinstance(key_values, Config):
        key_values = key_values.to_flatten_dict()

    if hasattr(key_values, 'items'):
        data = [(key, formatter(value)) for key, value in key_values.items()]
    else:
        data = [(key, formatter(value)) for key, value in key_values]

    # use the terminaltables.AsciiTable to format our key values
    table = AsciiTable(data)
    table.padding_left = 0
    table.padding_right = 3
    table.inner_column_border = False
    table.inner_footing_row_border = False
    table.inner_heading_row_border = False
    table.inner_row_border = False
    table.outer_border = False
    lines = [line.rstrip() for line in table.table.split('\n')]

    # prepend a title
    if title is not None:
        max_length = max(max(map(len, lines)), len(title))
        delim = delimiter_char * max_length
        lines = [title, delim] + lines

    return '\n'.join(lines)

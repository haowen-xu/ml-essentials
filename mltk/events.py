from typing import Callable, Dict, Iterator, Generic, TypeVar

__all__ = ['Event', 'EventHost']


TCallback = Callable[..., None]


class Event(object):
    """
    Event object, should only be constructed by :class:`EventHost`.
    """

    def __init__(self, host: 'EventHost', name: str):
        self._host = host
        self._name = name
        self._callbacks = []

    def __repr__(self):
        return f'Event({self._name})'

    def __call__(self, method: TCallback):
        """
        Register `method` as callback to this event.

        Args:
            method: The callback method.

        Returns:
            The specified `method`.
        """
        self.do(method)
        return method

    def do(self, callback: TCallback):
        """
        Register `callback` to this event.

        Args:
            callback: Callback to register.
        """
        self._callbacks.append(callback)

    def undo(self, callback: TCallback):
        """
        Unregister `callback` from this event.

        Args:
            callback: Callback to unregister.
                No error will be raised if it has not been registered yet.
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def fire(self, *args, **kwargs):
        """
        Fire this event.

        Args:
            *args: Positional arguments.
            **kwargs: Named arguments.
        """
        self._host.fire(self._name, *args, **kwargs)

    def reverse_fire(self, *args, **kwargs):
        """
        Fire this event, calling all callbacks in reversed order.

        Args:
            *args: Positional arguments.
            **kwargs: Named arguments.
        """
        self._host.reverse_fire(self._name, *args, **kwargs)


class EventHost(object):
    """
    Class to create and manage :class:`Event` objects.

    Callbacks can be registered to a event by :meth:`on()`, for example:

    >>> def print_args(*args, **kwargs):
    ...     print(args, kwargs)

    >>> events = EventHost()
    >>> events.on('updated', print_args)  # register `print_args` to a event
    >>> list(events)  # get names of created events
    ['updated']
    >>> 'updated' in events  # test whether a event name exists
    True
    >>> events.fire('updated', 123, second=456)  # fire the event
    (123,) {'second': 456}

    It is also possible to obtain an object that represents the event,
    and register callbacks / fire the event via that object, for example:

    >>> events = EventHost()
    >>> on_updated = events['updated']
    >>> list(events)
    ['updated']
    >>> on_updated
    Event(updated)
    >>> on_updated.do(print_args)
    >>> on_updated.fire(123, second=456)
    (123,) {'second': 456}
    """

    def __init__(self):
        self._connected_hosts = []
        self._events = {}  # type: Dict[str, Event]

    def __iter__(self) -> Iterator[str]:
        return iter(self._events)

    def __contains__(self, item) -> bool:
        return item in self._events

    def __getitem__(self, item) -> Event:
        if item not in self._events:
            self._events[item] = Event(self, item)
        return self._events[item]

    def connect(self, other: 'EventHost'):
        """
        Connect this event host with another event host, such that all events
        fired from this host will also be fired from that host.

        >>> events1 = EventHost()
        >>> events1.on('updated',
        ...     lambda *args, **kwargs: print('from events1', args, kwargs))
        >>> events2 = EventHost()
        >>> events2.on('updated',
        ...     lambda *args, **kwargs: print('from events2', args, kwargs))
        >>> events1.connect(events2)
        >>> events1.fire('updated', 123, second=456)
        from events1 (123,) {'second': 456}
        from events2 (123,) {'second': 456}

        Args:
            other: The other event host.
        """
        self._connected_hosts.append(other)

    def disconnect(self, other: 'EventHost'):
        """
        Disconnect this event host with another event host.

        Args:
            other: The other event host.

        See Also:
            :meth:`connect()`
        """
        if other in self._connected_hosts:
            self._connected_hosts.remove(other)

    def on(self, name: str, callback: Callable):
        """
        Register `callback` to an event.
        Args:
            name: Name of the event.
            callback: Callback to register.
        """
        self[name].do(callback)

    def off(self, name: str, callback: Callable):
        """
        Unregister `callback` from an event.

        Args:
            name: Name of the event.
            callback: Callback to unregister.
                No error will be raised if it has not been registered yet.
        """
        if name in self._events:
            self._events[name].undo(callback)

    def fire(self, name, *args, **kwargs):
        """
        Fire an event.

        Args:
            name: Name of the event.
            *args: Positional arguments.
            \\**kwargs: Named arguments.
        """
        event = self._events.get(name, None)
        if event is not None:
            for callback in event._callbacks:
                callback(*args, **kwargs)

            for host in self._connected_hosts:
                host.fire(name, *args, **kwargs)

    def reverse_fire(self, name, *args, **kwargs):
        """
        Fire an event, calling all callbacks in reversed order.

        Args:
            name: Name of the event.
            *args: Positional arguments.
            \\**kwargs: Named arguments.
        """
        event = self._events.get(name, None)
        if event is not None:
            for host in reversed(self._connected_hosts):
                host.reverse_fire(name, *args, **kwargs)

            for callback in reversed(event._callbacks):
                callback(*args, **kwargs)

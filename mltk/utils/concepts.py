__all__ = ['AutoInitAndCloseable']


class AutoInitAndCloseable(object):
    """
    Classes with :meth:`init()` to initialize its internal states, and also
    :meth:`close()` to destroy these states.  The :meth:`init()` method can
    be repeatedly called, which will cause initialization only at the first
    call.  Thus other methods may always call :meth:`init()` at beginning,
    which can bring auto-initialization to the class.

    A context manager is implemented: :meth:`init()` is explicitly called
    when entering the context, while :meth:`close()` is called when
    exiting the context.
    """

    _initialized = False

    def _init(self):
        """Override this method to initialize the internal states."""
        raise NotImplementedError()

    def init(self):
        """Ensure the internal states are initialized."""
        if not self._initialized:
            self._init()
            self._initialized = True

    def __enter__(self):
        """Ensure the internal states are initialized."""
        self.init()
        return self

    def _close(self):
        """Override this method to destroy the internal states."""
        raise NotImplementedError()

    def close(self):
        """Ensure the internal states are destroyed."""
        if self._initialized:
            try:
                self._close()
            finally:
                self._initialized = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup the internal states."""
        self.close()

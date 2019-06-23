import os

from .config import Config, ConfigField

__all__ = ['Settings', 'settings']


class Settings(Config):
    """Global settings of the whole `mltk` package."""

    cache_root: str = ConfigField(
        str,
        default=os.path.expanduser('~/.mltk/cache'),
        envvar='MLTK_CACHE_ROOT',
    )

    file_cache_checksum: bool = False
    """Whether or not to validate the checksum of cached files?"""


settings = Settings()

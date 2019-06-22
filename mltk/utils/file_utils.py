import os
import zipfile
from typing import *

__all__ = ['make_dir_archive']


def make_dir_archive(dir_path: str,
                     archive_path: str,
                     is_included: Callable[[str], bool] = None,
                     is_excluded: Callable[[str], bool] = None,
                     compression: int = zipfile.ZIP_STORED):
    """
    Pack all content of a directory into an archive.

    Args:
        dir_path: The source directory ptah.
        archive_path: The destination archive path.
        is_included: A callable ``(file_path) -> bool`` to check whether or not
            a file should be included in the archive.
        is_excluded: A callable ``(path) -> bool`` to check whether or not
            a directory or a file should be excluded in the archive.
        compression: The compression level.
    """
    if not os.path.isdir(dir_path):
        raise IOError(f'Not a directory: {dir_path}')

    def walk(path, relpath):
        for name in os.listdir(path):
            f_path = os.path.join(path, name)
            f_relpath = f'{relpath}/{name}' if relpath else name

            if is_excluded is not None and is_excluded(f_path):
                continue

            if os.path.isdir(f_path):
                zf.write(f_path, arcname=f_relpath)
                walk(f_path, f_relpath)
            elif is_included is None or is_included(f_path):
                zf.write(f_path, arcname=f_relpath)

    with zipfile.ZipFile(archive_path, 'w', compression=compression) as zf:
        walk(dir_path, '')

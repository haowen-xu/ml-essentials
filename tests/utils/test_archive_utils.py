import os
import re
import unittest
from tempfile import TemporaryDirectory

import pytest

from mltk.utils import make_dir_archive
from tests.helpers import prepare_dir, zip_snapshot


class ArchiveUtilsTestCase(unittest.TestCase):

    def test_make_dir_archive(self):
        with TemporaryDirectory() as temp_dir:
            source_dir = os.path.join(temp_dir, 'source')
            archive_path = os.path.join(temp_dir, 'archive.zip')

            with pytest.raises(IOError, match=f'Not a directory: {source_dir}'):
                make_dir_archive(source_dir, archive_path)

            # prepare for the source dir
            source_content = {
                'a.txt': b'a.txt',
                'a.exe': b'a.exe',
                'nested': {
                    'b.py': b'b.py',
                    '.DS_Store': b'.DS_Store',
                },
                '.git': {
                    'c.sh': b'c.sh',
                }
            }
            prepare_dir(source_dir, source_content)

            # test make archive without filters
            make_dir_archive(source_dir, archive_path)
            self.assertDictEqual(zip_snapshot(archive_path), source_content)

            # test make archive with filters
            archive_path = os.path.join(temp_dir, 'archive.zip')
            make_dir_archive(
                source_dir,
                archive_path,
                is_included=re.compile(r'.*(\.(txt|py|sh)|\.DS_Store)$').match,
                is_excluded=re.compile(r'.*(\.git|\.DS_Store)$').match,
            )
            self.assertDictEqual(zip_snapshot(archive_path), {
                'a.txt': b'a.txt',
                'nested': {
                    'b.py': b'b.py',
                }
            })

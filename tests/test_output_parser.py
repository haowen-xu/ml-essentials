# -*- coding: utf-8 -*-

import os
import re
import socket
import sys
import time
import unittest
from contextlib import contextmanager
from tempfile import TemporaryDirectory

import httpretty
import mock
import pytest
import requests
from bson import ObjectId
from click.testing import CliRunner
from mock import Mock

from mltk import Config, ConfigValidationError, MLStorageClient
from mltk.output_parser import *
from mltk.utils import json_dumps, json_loads
from tests.helpers import *



class ProgramOutputSinkTestCase(unittest.TestCase):

    def test_split_lines(self):
        class MyParser(ProgramOutputSink):
            def parse_line(self, line: bytes):
                logs.append(line)
                yield from super().parse_line(line)

        logs = []
        parser = MyParser(parsers=[])

        _ = list(parser.parse(b''))
        _ = list(parser.parse(b'no line break '))
        _ = list(parser.parse(b'until '))
        _ = list(parser.parse(b''))
        _ = list(parser.parse(b'this word\nanother line\nthen the third '))
        _ = list(parser.parse(b'line'))

        self.assertListEqual(logs, [
            b'no line break until this word',
            b'another line',
        ])
        _ = list(parser.flush())
        self.assertListEqual(logs, [
            b'no line break until this word',
            b'another line',
            b'then the third line',
        ])

        _ = list(parser.parse(b''))
        _ = list(parser.parse(b'the fourth line\n'))
        _ = list(parser.parse(b'the fifth line\n'))
        _ = list(parser.flush())
        self.assertListEqual(logs, [
            b'no line break until this word',
            b'another line',
            b'then the third line',
            b'the fourth line',
            b'the fifth line',
        ])

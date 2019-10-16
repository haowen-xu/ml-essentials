import re
from abc import ABC, abstractmethod
from typing import *
from dataclasses import dataclass

from dataclasses_json import dataclass_json

__all__ = [
    'ProgramStatus', 'ProgramProgress', 'ProgramWebUI',
    'ProgramOutputSink', 'ProgramLineParser', 'ProgramProgressParser',
    'ProgramWebUIParser',
]


@dataclass_json
@dataclass
class ProgramStatus(object):
    pass


@dataclass
class ProgramMetric(object):
    __slots__ = ('mean', 'std')

    mean: Optional[str]
    std: Optional[str]


@dataclass
class ProgramProgress(ProgramStatus):
    __slots__ = ('epoch', 'max_epoch', 'step', 'max_step',
                 'eta', 'metrics')

    epoch: Optional[int]
    max_epoch: Optional[int]
    step: Optional[int]
    max_step: Optional[int]
    eta: Optional[str]
    metrics: Dict[str, ProgramMetric]


@dataclass
class ProgramWebUI(ProgramStatus):
    __slots__ = ('name', 'uri')

    name: str
    uri: str


class ProgramOutputSink(object):
    """Sink to accumulate output lines, and send to line parsers."""

    def __init__(self,
                 parsers: Sequence['ProgramLineParser'],
                 max_parse_length: int = 2048):
        self._parsers: List[ProgramLineParser] = list(parsers)
        self._max_parse_length: int = max_parse_length
        self._line_buffer: Optional[bytes] = None

    def parse_line(self, line: bytes) -> Generator[ProgramStatus, None, None]:
        if len(line) <= self._max_parse_length:
            for parser in self._parsers:
                count = 0
                for status in parser.parse_line(line):
                    count += 1
                    yield status
                if count > 0:
                    # already matched by one parser, stop propagation
                    break

    def parse(self, content: bytes) -> Generator[ProgramStatus, None, None]:
        """
        Parse the program output.

        Args:
            content: The output content.

        Yields:
            The program status objects.
        """
        # find the first line break
        start = 0
        end = content.find(b'\n')
        if end != -1:
            if self._line_buffer:
                yield from self.parse_line(self._line_buffer + content[: end])
                self._line_buffer = None
            else:
                yield from self.parse_line(content[: end])
            start = end + 1

            while start < len(content):
                end = content.find(b'\n', start)
                if end != -1:
                    yield from self.parse_line(content[start: end])
                    start = end + 1
                else:
                    break

        if start < len(content):
            if self._line_buffer:
                self._line_buffer = self._line_buffer + content[start:]
            else:
                self._line_buffer = content[start:]

    def flush(self) -> Generator[ProgramStatus, None, None]:
        """
        Parse the un-parsed content as a complete line.

        Yields:
            The program status objects.
        """
        if self._line_buffer:
            yield from self.parse_line(self._line_buffer)
            self._line_buffer = None


class ProgramLineParser(ABC):
    """Base class for parsing each line of the program output."""

    @abstractmethod
    def parse_line(self, line: bytes) -> Generator[ProgramStatus, None, None]:
        """
        Parse the given program output line, and get all status objects.

        Args:
            line: The line content.

        Yields:
            The program status objects.
        """
        pass  # pragma: no cover


class ProgramProgressParser(ProgramLineParser):

    DEFAULT_LINE_PATTERN = (
        rb'^\['
        rb'(?:Epoch (?P<epoch>\d+)(?:/(?P<max_epoch>\d+))?)?[, ]*'
        rb'(?:Step (?P<step>\d+)(?:/(?P<max_step>\d+))?)?[, ]*'
        rb'(?:ETA (?P<eta>[0-9\.e+ dhms]+))?'
        rb'\]\s*'
        rb'(?P<metrics>.*?)\s*(?:\(\*\))?\s*'
        rb'$'
    )
    DEFAULT_METRIC_PATTERN = (
        rb'^\s*(?P<name>[^:]+): (?P<mean>[^()]+)'
        rb'(?: \(\xc2\xb1(?P<std>[^()]+)\))?\s*$'
    )

    def __init__(self,
                 line_pattern: Optional[bytes] = None,
                 metric_pattern: Optional[bytes] = None):
        if line_pattern is None:
            line_pattern = self.DEFAULT_LINE_PATTERN
        if metric_pattern is None:
            metric_pattern = self.DEFAULT_METRIC_PATTERN

        self._line_pattern = re.compile(line_pattern)
        self._metric_pattern = re.compile(metric_pattern)

    def parse_line(self, line: bytes) -> Generator[ProgramStatus, None, None]:
        m = self._line_pattern.match(line)
        if m:
            g = m.groupdict()

            # the progress
            progress = {
                'epoch': None,
                'max_epoch': None,
                'step': None,
                'max_step': None
            }
            for key in progress.keys():
                if g.get(key, None) is not None:
                    progress[key] = int(g[key])

            progress['eta'] = None
            if g.get('eta', None) is not None:
                progress['eta'] = g['eta'].decode('utf-8').strip()

            # the metrics
            metrics = {}
            metric_pieces = g.pop('metrics', None)

            if metric_pieces:
                metric_pieces = metric_pieces.split(b';')
                for metric in metric_pieces:
                    m = self._metric_pattern.match(metric)
                    if m:
                        g = m.groupdict()
                        name = g['name'].decode('utf-8').strip()
                        mean = g['mean'].decode('utf-8').strip()
                        if g.get('std', None) is not None:
                            std = g['std'].decode('utf-8').strip()
                        else:
                            std = None

                        # special hack: tfsnippet replaced "_" by " ",
                        # but we now do not use this replacement.
                        name = name.replace(' ', '_')
                        metrics[name] = (mean, std)

            # filter out none items
            metrics = {
                k: ProgramMetric(mean=v[0], std=v[1])
                for k, v in metrics.items()
                if v is not None
            }

            # now trigger the event
            yield ProgramProgress(metrics=metrics, **progress)


class ProgramWebUIParser(ProgramLineParser):

    DEFAULT_PATTERN = (
        rb'(?:^TensorBoard \S+ at (?P<TensorBoard>\S+))|'
        rb'(?:^Serving HTTP on \S+ port \d+ \((?P<SimpleHTTP>[^()]+)\))'
    )

    def __init__(self, pattern: Optional[bytes] = None):
        if pattern is None:
            pattern = self.DEFAULT_PATTERN
        self._pattern = re.compile(pattern)

    def parse_line(self, line: bytes) -> Generator[ProgramStatus, None, None]:
        m = self._pattern.match(line)
        if m:
            g = m.groupdict()
            for key, val in g.items():
                if val is not None:
                    uri = val.decode('utf-8')
                    yield ProgramWebUI(name=key, uri=uri)

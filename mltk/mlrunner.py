# -*- coding: utf-8 -*-
import codecs
import copy
import logging
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import time
import traceback
import typing
import zipfile
from contextlib import contextmanager, ExitStack
from logging import getLogger, FileHandler
from threading import RLock, Condition, Thread
from typing import *

import click

from .config import Config, ConfigLoader
from .events import EventHost, Event
from .mlstorage import DocumentType, MLStorageClient, IdType, json_loads
from .utils import exec_proc

__all__ = ['MLRunnerConfig', 'MLRunner']

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s [%(levelname)-8s] %(message)s'

PatternType = getattr(typing, 'Pattern', getattr(re, 'Pattern', Any))


class MLRunnerConfig(Config):
    """
    Config for :class:`MLRunner`.

    >>> config = MLRunnerConfig(server='http://127.0.0.1:8080',
    ...                         args=['python', 'run.py'])
    >>> config.tags = ['hello', 123]
    >>> config.env = {'THE_VAR': 456}
    >>> config.gpu = ['1', '2']
    >>> config.daemon = ['echo "hello"', ['python', '-m', 'http.server']]

    >>> config = config.validate()
    >>> config.server
    'http://127.0.0.1:8080'
    >>> config.args
    ['python', 'run.py']
    >>> config.tags
    ['hello', '123']
    >>> config.env
    Config(THE_VAR='456')
    >>> config.gpu
    [1, 2]
    >>> config.daemon
    ['echo "hello"', ['python', '-m', 'http.server']]
    """

    server: str = None
    name: str = None
    description: str = None
    tags: List[str] = None
    resume_from: str = None
    clone_from: str = None

    args: Union[str, List[str]] = None
    daemon: List[List[str]] = None
    work_dir: str = None
    env: Config = None
    gpu: List[int] = None

    quiet: bool = False

    class source(Config):
        root: str = '.'
        clone_dir: bool = False
        cleanup: bool = True
        pack_zip: bool = True
        source_zip_file: str = 'source.zip'

        includes = [
            re.compile(r'.*\.(py|pl|rb|js|sh|r|bat|cmd|exe|jar)$', re.I)
        ]
        excludes = [
            re.compile(
                r'.*[\\/](node_modules|\.svn|\.cvs|\.idea|'
                r'\.DS_Store|\.git|\.hg|\.pytest_cache|__pycache__)$',
                re.I
            )
        ]

        def user_validate(self):
            def maybe_compile(p):
                if not hasattr(p, 'match'):
                    p = re.compile(p, re.I)
                return p

            if self.includes:
                if not isinstance(self.includes, (list, tuple)):
                    self.includes = [self.includes]
                self.includes = [maybe_compile(p) for p in self.includes]

            if self.excludes:
                if not isinstance(self.excludes, (list, tuple)):
                    self.excludes = [self.excludes]
                self.excludes = [maybe_compile(p) for p in self.excludes]

    class integration(Config):
        parse_stdout: bool = True

        log_file: str = 'console.log'
        daemon_log_file: str = 'daemon.log'
        runner_log_file: str = 'mlrun.log'

        config_file: str = 'config.json'
        default_config_file: str = 'config.defaults.json'
        result_file: str = 'result.json'
        webui_file: str = 'webui.json'

    def user_validate(self):
        if self.server is None:
            raise ValueError('`server` is required.')

        if self.args is None:
            raise ValueError('`args` is required.')
        elif not isinstance(self.args, (str, bytes)):
            self.args = list(map(str, self.args))
        else:
            self.args = str(self.args)

        if not self.args:
            raise ValueError('`args` cannot be empty.')

        if self.tags is not None:
            if not isinstance(self.tags, (list, tuple)):
                self.tags = [self.tags]
            self.tags = list(map(str, self.tags))

        if self.env is not None:
            self.env = Config(**{
                key: str(value)
                for key, value in self.env.to_dict().items()
            })

        if self.gpu is not None:
            if not isinstance(self.gpu, (list, tuple)):
                self.gpu = [self.gpu]
            self.gpu = list(map(int, self.gpu))

        if self.daemon is not None:
            if not isinstance(self.daemon, (list, tuple)):
                raise ValueError(f'`daemon` must be a sequence: '
                                 f'got {self.daemon!r}.')

            daemon = []
            for d in self.daemon:
                if not isinstance(d, (list, tuple)):
                    d = str(d)
                else:
                    d = list(map(str, d))
                daemon.append(d)
            self.daemon = daemon


class MLRunner(object):

    def __init__(self, config: MLRunnerConfig):
        self._config = config
        self._client = MLStorageClient(config.server)
        self._doc = None  # type: ExperimentDoc
        self._clone_doc = None  # type: DocumentType

    @property
    def config(self) -> MLRunnerConfig:
        return self._config

    @property
    def client(self) -> MLStorageClient:
        return self._client

    @property
    def doc(self) -> 'ExperimentDoc':
        return self._doc

    def env_dict(self, output_dir, work_dir) -> Dict[str, str]:
        """
        Get the environment dict for this process.

        Args:
            output_dir: The MLStorage output directory.
            work_dir: The work directory.
        """
        def update_env(target, source):
            if source:
                target.update(source)

        env = os.environ.copy()

        # runner default environment variables
        update_env(env, {
            'PYTHONUNBUFFERED': '1',
            'MLRUNNER_OUTPUT_DIR': output_dir,
            'MLSTORAGE_SERVER': self.config.server,
        })
        update_env(env, self.config.env.to_flatten_dict())

        # set "CUDA_VISIBLE_DEVICES" according to gpu settings
        if self.config.gpu:
            update_env(env, {
                'CUDA_VISIBLE_DEVICES': ','.join(map(str, self.config.gpu))
            })

        # set "PWD" to work_dir
        update_env(env, {'PWD': work_dir})

        # add ".", output_dir and work_dir to python path
        new_python_path = [
            os.path.abspath('.'),
            os.path.abspath(output_dir),
            os.path.abspath(work_dir),
        ]
        if env.get('PYTHONPATH'):
            new_python_path.insert(0, env.get('PYTHONPATH'))
        env['PYTHONPATH'] = os.pathsep.join(new_python_path)

        return env

    def _copy_dir(self, src, dst):
        os.makedirs(dst, exist_ok=True)
        for name in os.listdir(src):
            src_path = os.path.join(src, name)
            dst_path = os.path.join(dst, name)
            if os.path.isdir(src_path):
                self._copy_dir(src_path, dst_path)
            else:
                shutil.copyfile(src_path, dst_path, follow_symlinks=False)

    def _fs_stats(self, path):
        try:
            st = os.stat(path, follow_symlinks=False)
        except IOError:
            return 0, 0
        else:
            size, count = st.st_size, 1
            if stat.S_ISDIR(st.st_mode):
                for name in os.listdir(path):
                    f_size, f_count = self._fs_stats(os.path.join(path, name))
                    size += f_size
                    count += f_count
            return size, count

    def _on_mltk_log(self, progress, metrics):
        updates = {}

        if metrics:
            def format_value(v):
                if isinstance(v, (tuple, list)):
                    if v[1] is not None:
                        return f'{v[0]} ± {v[1]}'
                    else:
                        return v[0]
                else:
                    return v

            metrics = {k: format_value(v) for k, v in metrics.items()
                       if k.rsplit('_', 1)[-1] != 'time'}

            if metrics:
                updates['result'] = metrics

        if progress:
            def set_counter(name, v, max_v):
                if v is not None:
                    if max_v is not None:
                        tmp[name] = f'{v}/{max_v}'
                    else:
                        tmp[name] = v

            tmp = {}
            set_counter('epoch', progress.get('epoch'),
                        progress.get('max_epoch'))
            set_counter('step', progress.get('step'),
                        progress.get('max_step'))
            if progress.get('eta'):
                tmp['eta'] = progress['eta']

            if tmp:
                updates['progress'] = tmp

        if updates:
            self.doc.update(updates)

    def _on_webui_log(self, info):
        if info:
            self.doc.update({'webui': info})

    def _create_log_parser(self):
        parser = StdoutParser()
        parser.on_mltk_log.do(self._on_mltk_log)
        parser.on_webui_log.do(self._on_webui_log)
        return parser

    def _on_json_updated(self, name, content):
        if name == self.config.integration.result_file:
            self.doc.update({'result': content})
        elif name == self.config.integration.config_file:
            self.doc.update({'config': content})
        elif name == self.config.integration.default_config_file:
            self.doc.update({'default_config': content})
        elif name == self.config.integration.webui_file:
            self.doc.update({'webui': content})

    def _run_proc(self, output_dir):
        # if work_dir is not set, use the MLStorage output_dir (storage_dir).
        if self.config.work_dir is not None:
            work_dir = os.path.abspath(self.config.work_dir)
        else:
            work_dir = output_dir

        # prepare for the environment variable
        env = self.env_dict(output_dir, work_dir)

        # update the doc with execution info
        self.doc.update({
            'args': self.config.args,
            'exc_info': {
                'hostname': socket.gethostname(),
                'pid': os.getpid(),
                'env': env,
            }
        })

        # create the program host for the main process
        log_file = os.path.join(output_dir, self.config.integration.log_file)
        main_proc = ProgramHost(
            args=self.config.args,
            env=env,
            work_dir=work_dir,
            log_to_stdout=not self.config.quiet,
            log_parser=self._create_log_parser(),
            log_file=log_file,
            append_to_file=True,
        )

        # create the program hosts for the daemon processes
        daemons = []
        if self.config.daemon:
            for i, daemon in enumerate(self.config.daemon):
                a, b = os.path.splitext(self.config.integration.daemon_log_file)
                log_file = os.path.join(output_dir, f'{a}.{i}{b}')
                daemons.append(ProgramHost(
                    args=daemon,
                    env=env,
                    work_dir=work_dir,
                    log_to_stdout=False,
                    log_parser=self._create_log_parser(),
                    log_file=log_file,
                    append_to_file=True,
                ))

        # populate the work directory and output dir
        if self._clone_doc:
            self._copy_dir(self._clone_doc['storage_dir'], output_dir)

        source_copier = SourceCopier(
            source_dir='.',
            dest_dir=output_dir,
            includes=self.config.source.includes,
            excludes=self.config.source.excludes
        )

        # create the background json watcher
        json_watcher = JsonFileWatcher(
            root_dir=output_dir,
            file_names=[
                self.config.integration.result_file,
                self.config.integration.config_file,
                self.config.integration.default_config_file,
                self.config.integration.webui_file,
            ]
        )
        json_watcher.on_json_updated.do(self._on_json_updated)

        # now execute the processes
        with ExitStack() as ctx_stack:
            # populate the work directory with source files
            if self.config.source.clone_dir:
                ctx_stack.enter_context(source_copier)
                getLogger(__name__).info(
                    'Cloned source files to work directory.')

            if self.config.source.pack_zip:
                source_zip_file = os.path.join(
                    output_dir, self.config.source.source_zip_file)
                source_copier.pack_zip(source_zip_file)
                getLogger(__name__).info(
                    'Created source file archive: %s', source_zip_file)

            # start the background json watcher
            ctx_stack.enter_context(json_watcher)

            # start the daemon processes
            for daemon, daemon_args in zip(daemons, self.config.daemon):
                daemon_proc = ctx_stack.enter_context(daemon.exec_proc())
                getLogger(__name__).info(
                    'Started daemon process %s: %s',
                    daemon_proc.pid, daemon_args
                )

            # run the main process
            with main_proc.exec_proc() as proc:
                getLogger(__name__).info(
                    'Started experiment process %s: %s',
                    proc.pid, self.config.args
                )
                code = proc.wait()
                getLogger(__name__).info(
                    f'Experiment process exited with code: {code}')

            return proc.wait()

    def run(self):
        # get the previous experiment if we are cloning from any one
        if self.config.clone_from:
            self._clone_doc = self.client.get(self.config.clone_from)

        # create an experiment, or resume from the previous experiment
        meta = {}
        for key in ('name', 'description', 'tags'):
            if self.config[key] is not None:
                meta[key] = self.config[key]

        # name is required on some version, so if it is not specified,
        # generate one
        if 'name' not in meta:
            if isinstance(self.config.args, (str, bytes)):
                meta['name'] = self.config.args
            else:
                meta['name'] = ' '.join(map(shlex.quote, self.config.args))

        if self.config.resume_from:
            doc = self.client.get(self.config.resume_from)
        else:
            doc = self.client.create(meta)

        # declare the variables to be used across all parts of try block
        fs_size = None
        inode_count = None
        exit_code = None
        doc_id = doc['id']

        try:
            output_dir = doc['storage_dir']
            os.makedirs(output_dir, exist_ok=True)

            # function to help filter out None dict items
            def filter_dict(d):
                return {k: v for k, v in d.items() if v is not None}

            # prepare for the work dir
            runner_log_file = os.path.join(
                output_dir, self.config.integration.runner_log_file)

            with configure_logger(runner_log_file):
                try:
                    # create the doc object to manage further updates
                    self._doc = ExperimentDoc(self.client, doc)
                    self.doc.start_worker()

                    try:
                        exit_code = self._run_proc(output_dir)
                    finally:
                        fs_size, inode_count = self._fs_stats(output_dir)

                except Exception as ex:
                    final_status = 'FAILED'
                    self.doc.set_finished(final_status, filter_dict({
                        'error': {
                            'message': str(ex),
                            'traceback': ''.join(
                                traceback.format_exception(*sys.exc_info()))
                        },
                        'exit_code': exit_code,
                        'storage_size': fs_size,
                        'storage_inode': inode_count,
                    }))
                    getLogger(__name__).error(
                        'Failed to run the experiment.', exc_info=True)
                else:
                    final_status = 'COMPLETED'
                    self.doc.set_finished(final_status, filter_dict({
                        'exit_code': exit_code,
                        'storage_size': fs_size,
                        'storage_inode': inode_count,
                    }))
                finally:
                    self.doc.stop_worker()
                    remain_updates = self.doc.merge_doc_updates()

                    if remain_updates:  # pragma: no cover
                        # final attempt to save the remaining updates
                        # should rarely happen
                        self.client.update(doc_id, remain_updates)

        except Exception as ex:
            self.client.set_finished(doc_id, 'FAILED', {
                'error': {
                    'message': str(ex),
                    'traceback': ''.join(
                        traceback.format_exception(*sys.exc_info()))
                }
            })
            getLogger(__name__).error(
                'Failed to run the experiment.', exc_info=True)

        return exit_code


@click.command()
@click.option('-n', '--name', required=False, default=None,
              help='Experiment name.')
@click.option('-d', '--description', required=False, default=None,
              help='Experiment description.')
@click.option('-t', '--tags', required=False, multiple=True,
              help='Experiment tags, comma separated strings, e.g. '
                   '"prec 0.996, state of the arts".')
@click.option('-C', '--config-file', required=False, multiple=True,
              help='Load runner configuration from JSON or YAML file. '
                   'Values from all config files will be merged. '
                   'The CLI arguments will override all config files.')
@click.option('-e', '--env', required=False, multiple=True,
              help='Environmental variable (FOO=BAR).')
@click.option('--gpu', required=False, multiple=True,
              help='Quick approach to set the "CUDA_VISIBLE_DEVICES" '
                   'environmental variable.')
@click.option('-s', '--server', required=False,
              default=os.environ.get('MLSTORAGE_SERVER', '') or None,
              help='Specify the URI of MLStorage API server, e.g., '
                   '"http://localhost:8080".  If not specified, will use '
                   '``os.environ["MLSTORAGE_SERVER"]``.')
@click.option('--clone/--no-clone',  'clone_source', is_flag=True, default=None,
              required=False,
              help='Whether or not to clone the source files from current '
                   'directory to MLStorage output directory?')
@click.option('--pack-zip/--no-pack-zip', 'pack_zip', is_flag=True,
              default=None, required=False,
              help='Whether or not to pack the source files from the current '
                   'directory into a zip archive in the MLStorage output '
                   'directory?')
@click.option('-D', '--daemon', required=False, multiple=True,
              help='Specify the shell command of daemon processes, to be '
                   'executed along with the main experiment process.')
@click.option('-c', 'command', required=False, default=None,
              help='Specify the shell command to execute. '
                   'Will override the program arguments (args).')
@click.argument('args', nargs=-1)
def mlrun(name, description, tags, config_file, env, gpu, server, clone_source,
          pack_zip, daemon, command, args):
    """
    Run an experiment.

    The program arguments should be either specified at the end, after a "--"
    mark, or specified as command line via "-c" argument.  For example::

        mlrun -- python train.py
        mlrun -c "python train.py"

    By default, the program will not run in the current directory.
    Instead, it will run in the MLStorage output directory, assigned by the
    server.  If you need to copy some source files to the output directory,
    you may specify the "--clone" argument, for example::

        mlrun --clone -- python train.py

    The following file extensions will be regarded as source files::

        *.py *.pl *.rb *.js *.sh *.r *.bat *.cmd *.exe *.jar

    The source files will also be collected into
    "<MLStorage output dir>/source.zip", unless you specify "--no-pack-zip".

    During the execution of the program, the STDOUT and STDERR of the program
    will be captured and stored in "<MLStorage output dir>/console.log".

    The program may write config values as JSON into
    "<MLStorage output dir>/config.json", default config values as JSON into
    "<MLStorage output dir>/config.defaults.json", results as JSON into
    "<MLStorage output dir>/result.json", and exposed web services into
    "<MLStorage output dir>/webui.json".
    The runner will collect objects from these JSON files, and save to the
    MLStorage server.

    An example of "config.defaults.json"::

        {"max_epoch": 1000, "learning_rate": 0.01}

    An example of "result.json"::

        {"accuracy": 0.996}

    And an example of "webui.json"::

        {"TensorBoard": "http://[ip]:6006"}

    The layout of the experiment storage directory is shown as follows::

    \b
        .
        |-- config.json
        |-- config.defaults.json
        |-- console.log
        |-- daemnon.X.log
        |-- mlrun.log
        |-- result.json
        |-- source.zip
        |-- webui.json
        |-- ... (cloned source files)
        `-- ... (other files generated by the program)

    After the execution of the program, the cloned source files will be deleted
    from the MLStorage output directory.
    """
    logging.basicConfig(level='INFO', format=LOG_FORMAT)

    # load configuration from files
    config_loader = MLRunnerConfigLoader(config_files=config_file)
    config_loader.load_config_files(
        lambda path: getLogger(__name__).info(
            'Load runner configuration from: %s', path)
    )

    # parse the CLI arguments
    if env:
        env_dict = {}
        for e in env:
            k, v = e.split('=', 1)
            env_dict[k] = v
    else:
        env_dict = None

    if gpu:
        gpu_list = []
        for g in gpu:
            gpu_list.extend(filter(
                (lambda s: s),
                (s.strip() for s in g.split(','))
            ))
    else:
        gpu_list = None

    # feed CLI arguments into MLRunnerConfig
    cli_config = {
        'name': name,
        'description': description,
        'tags': tags,
        'env': env_dict,
        'gpu': gpu_list,
        'server': server,
        'source.clone_dir': clone_source,
        'source.pack_zip': pack_zip,
        'daemon': daemon,
        'args': command or args,
    }
    cli_config = {k: v for k, v in cli_config.items()
                  if v is not None}

    config_loader.load_object(cli_config)
    config = config_loader.get()

    # now create the runner and run
    runner = MLRunner(config)
    exit_code = runner.run()

    if exit_code is not None:
        sys.exit(exit_code)
    else:
        sys.exit(-1)


@contextmanager
def configure_logger(log_file: str):
    """
    Open a context that captures the logs of MLRunner into specified file.

    Args:
        log_file: Path of the log file.
    """
    logger = getLogger(__name__)
    level = logger.level
    fmt = logging.Formatter(LOG_FORMAT)

    f_handler = FileHandler(log_file)
    f_handler.setLevel(LOG_LEVEL)
    f_handler.setFormatter(fmt)

    try:
        logger.addHandler(f_handler)
        logger.level = getattr(logging, LOG_LEVEL)
        yield
    finally:
        logger.removeHandler(f_handler)
        logger.level = level
        f_handler.close()


class MLRunnerConfigLoader(ConfigLoader[MLRunnerConfig]):
    """
    The config loader for :class:`MLRunnerConfig`.
    """

    def __init__(self,
                 config: Optional[MLRunnerConfig] = None,
                 config_files: Optional[Iterable[str]] = None,
                 work_dir: Optional[str] = None,
                 system_paths: Iterable[str] = (os.path.expanduser('~'),),
                 file_names: Iterable[str] = ('.mlrun.yml', '.mlrun.yaml',
                                              '.mlrun.json')):
        """
        Construct a new :class:`MLRunnerConfigLoader`.

        Args:
            config: The partial config object.
            config_files: User specified config files to load.
            work_dir: The work directory.  If specified, from the work
                directory, and from its all parents, config files will be
                searched.
            system_paths: The system paths, from where to search config files.
        """
        if config is None:
            config = MLRunnerConfig()
        if config_files is not None:
            config_files = tuple(map(str, config_files))
        if work_dir is not None:
            work_dir = os.path.abspath(work_dir)
        system_paths = tuple(map(os.path.abspath, system_paths))
        file_names = tuple(map(str, file_names))

        super().__init__(config, validate_all=True)
        self._user_config_files = config_files
        self._work_dir = work_dir
        self._system_paths = system_paths
        self._file_names = file_names

    def list_config_files(self) -> List[str]:
        """
        List all existing config files from search paths.

        The config files are ordered in ASCENDING priority, that is, you
        may use a :class:`MLRunnerConfigLoader` to load them in order.
        """
        files = []

        def try_add(f_path):
            if os.path.isfile(f_path) and f_path not in files:
                files.append(f_path)

        # check the user files
        if self._user_config_files:
            for f_path in reversed(self._user_config_files):
                try_add(f_path)

        # check the work directory
        if self._work_dir:
            work_dir = self._work_dir
            while True:
                for name in reversed(self._file_names):
                    try_add(os.path.join(work_dir, name))
                parent_dir = os.path.dirname(work_dir)
                if parent_dir == work_dir:
                    break
                work_dir = parent_dir

        # search for the system paths
        for path in reversed(self._system_paths):
            for name in reversed(self._file_names):
                try_add(os.path.join(path, name))

        # now compose the final list
        files.reverse()
        return files

    def load_config_files(self,
                          on_load: Optional[Callable[[str], None]] = None):
        """
        Load all config files returned by :meth:`list_config_files()`.

        Args:
            on_load: Callback function when a config file is being loaded.
        """
        for config_file in self.list_config_files():
            if on_load is not None:
                on_load(config_file)
            self.load_file(config_file)


class StdoutParser(object):
    """
    Class to parse the stdout of an experiment program.

    >>> parser = StdoutParser()
    >>> parser.on_mltk_log.do(print)
    >>> parser.on_webui_log.do(print)

    Various outputs can be parsed and the corresponding events can be triggered:

    >>> parser.parse_line('[Epoch 1/10, Step 5, ETA 2h 10s] '
    ...                   'epoch time: 3.2s; step time: 0.1s (±0.02s)'.
    ...                   encode('utf-8'))
    {'epoch': 1, 'max_epoch': 10, 'step': 5, 'eta': '2h 10s'} {'epoch_time': ('3.2s', None), 'step_time': ('0.1s', '0.02s')}

    >>> parser.parse_line(b'TensorBoard 1.13.1 at http://127.0.0.1:62462 '
    ...                   b'(Press CTRL+C to quit)')
    {'TensorBoard': 'http://127.0.0.1:62462'}
    >>> parser.parse_line(b'Serving HTTP on 0.0.0.0 port 8000 '
    ...                   b'(http://0.0.0.0:8000/) ...')
    {'Python HTTP Server': 'http://0.0.0.0:8000/'}

    >>> parser.parse_line(b'no pattern exist')

    Too long output lines will not be parsed, for example:

    >>> parser.parse_line(b'[Epoch 1/10, Step 5] ')
    {'epoch': 1, 'max_epoch': 10, 'step': 5} {}
    >>> parser.parse_line(b'[Epoch 1/10, Step 5]' + b' ' * 2048)
    """

    def __init__(self, max_parse_length: int = 2048):
        """
        Construct a new :class:`StdoutParser`.

        Args:
            max_parse_length: The maximum length of line to parse.
        """
        self._max_parse_length = max_parse_length
        self._events = EventHost()
        self._on_mltk_log = self.events['on_mltk_log']
        self._on_webui_log = self.events['on_webui_log']

        # buffer for accumulating lines
        self._line_buffer = None

        # patterns for parsing tfsnippet & mltk logs
        self._mltk_pattern = re.compile(
            rb'^\['
            rb'(?:Epoch (?P<epoch>\d+)(?:/(?P<max_epoch>\d+))?)?[, ]*'
            rb'(?:Step (?P<step>\d+)(?:/(?P<max_step>\d+))?)?[, ]*'
            rb'(?:ETA (?P<eta>[0-9\.e+ dhms]+))?'
            rb'\]\s*'
            rb'(?P<metrics>.*?)\s*(?:\(\*\))?\s*'
            rb'$'
        )
        self._mltk_metric_pattern = re.compile(
            rb'^\s*(?P<name>[^:]+): (?P<mean>[^()]+)'
            rb'(?: \(\xc2\xb1(?P<std>[^()]+)\))?\s*$'
        )

        # pattern for parsing tensorboard log
        self._webui_pattern = re.compile(
            rb'(?:^TensorBoard \S+ at (?P<TensorBoard>\S+))|'
            rb'(?:^Serving HTTP on \S+ port \d+ \((?P<PythonHTTP>[^()]+)\))',
            re.I
        )
        self._webui_keys = {
            'TensorBoard': 'TensorBoard',
            'PythonHTTP': 'Python HTTP Server',
        }

    @property
    def max_parse_length(self):
        """Get the maximum length of line to parse."""
        return self._max_parse_length

    @property
    def events(self) -> EventHost:
        """Get the event host."""
        return self._events

    @property
    def on_mltk_log(self) -> Event:
        """
        The event that an MLTK log has been received.

        Callback function type: `(progress: dict, metrics: dict) -> None`
        """
        return self._on_mltk_log

    @property
    def on_webui_log(self) -> Event:
        """
        The event that a Web UI has been created.

        Callback function type: `(info: dict) -> None`,
        where the structure of `info` is: `{'<name>': '<uri>'}`.
        """
        return self._on_webui_log

    def parse_line(self, line: bytes):
        """
        Parse an Stdout line.

        Args:
            line: The line content.
        """
        # check the length limit
        if len(line) > self.max_parse_length:
            return

        # try parse as MLTK log
        m = self._mltk_pattern.match(line)
        if m:
            g = m.groupdict()

            # the progress
            progress = {}
            for key in ('epoch', 'max_epoch', 'step', 'max_step'):
                if g[key] is not None:
                    progress[key] = int(g[key])
            if g['eta'] is not None:
                progress['eta'] = g['eta'].decode('utf-8').strip()

            # the metrics
            metrics = {}
            metric_pieces = g.pop('metrics', None)

            if metric_pieces:
                metric_pieces = metric_pieces.split(b';')
                for metric in metric_pieces:
                    m = self._mltk_metric_pattern.match(metric)
                    if m:
                        g = m.groupdict()
                        name = g['name'].decode('utf-8').strip()
                        mean = g['mean'].decode('utf-8').strip()
                        if g['std'] is not None:
                            std = g['std'].decode('utf-8').strip()
                        else:
                            std = None

                        # special hack: tfsnippet replaced "_" by " ",
                        # but we now do not use this replacement.
                        name = name.replace(' ', '_')
                        metrics[name] = (mean, std)

            # filter out none items
            metrics = {k: v for k, v in metrics.items() if v is not None}

            # now trigger the event
            self.on_mltk_log.fire(progress, metrics)
            return

        # try parse as TensorBoard logs
        m = self._webui_pattern.match(line)
        if m:
            g = m.groupdict()
            for key, val in self._webui_keys.items():
                if g[key] is not None:
                    self.on_webui_log.fire({val: g[key].decode('utf-8')})
            return

    def parse(self, content: bytes):
        """
        Parse the output content.

        Args:
            content: The output content.
        """
        # find the first line break
        start = 0
        end = content.find(b'\n')
        if end != -1:
            if self._line_buffer:
                self.parse_line(self._line_buffer + content[: end])
                self._line_buffer = None
            else:
                self.parse_line(content[: end])
            start = end + 1

            while start < len(content):
                end = content.find(b'\n', start)
                if end != -1:
                    self.parse_line(content[start: end])
                    start = end + 1
                else:
                    break

        if start < len(content):
            if self._line_buffer:
                self._line_buffer = self._line_buffer + content[start:]
            else:
                self._line_buffer = content[start:]

    def flush(self):
        """
        Parse the un-parsed content as a complete line.
        """
        if self._line_buffer:
            self.parse_line(self._line_buffer)
            self._line_buffer = None


class ProgramHost(object):
    """
    Class to run a program.
    """

    def __init__(self,
                 args: Union[str, List[str]],
                 env: Optional[Dict[str, Any]] = None,
                 work_dir: Optional[str] = None,
                 log_to_stdout: bool = True,
                 log_parser: Optional[StdoutParser] = None,
                 log_file: Optional[Union[str, int]] = None,
                 append_to_file: bool = True):
        """
        Construct a new :class:`ProgramHost`.

        Args:
            args: The program to execute, a command line str or a list of
                arguments str.  If it is a command line str, it will be
                executed by shell.
            env: The environment dict.
            work_dir: The working directory.
            log_to_stdout: Whether or not to write the program outputs to
                the runner's stdout?
            log_parser: The log parser, to parse the program outputs.
            log_file: The path or fileno of the log file, where to write the
                program outputs.
            append_to_file: Whether or not to open the log file in append mode?
        """
        self._args = args
        self._env = env
        self._work_dir = work_dir
        self._log_to_stdout = log_to_stdout
        self._log_parser = log_parser
        self._log_file = log_file
        self._append_to_file = append_to_file

    @contextmanager
    def exec_proc(self) -> Generator[subprocess.Popen, None, None]:
        """Run the program, and yield the process object."""
        # prepare for the arguments
        args = self._args

        # prepare for the environment dict
        env = self._env

        # prepare for the stdout duplicator
        if self._log_to_stdout:
            stdout_fileno = sys.stdout.fileno()

            def write_to_stdout(cnt):
                os.write(stdout_fileno, cnt)

        else:
            write_to_stdout = None

        # prepare for the log parser
        if self._log_parser is not None:
            def parse_log(cnt):
                try:
                    self._log_parser.parse(cnt)
                except Exception as ex:
                    getLogger(__name__).warning(
                        'Error in parsing output of: %s', args, exc_info=True)
        else:
            parse_log = None

        # prepare for the log file
        if self._log_file is not None:
            if isinstance(self._log_file, int):
                log_fileno = self._log_file
                close_log_file = False
            else:
                open_mode = os.O_WRONLY | os.O_CREAT
                if self._append_to_file:
                    open_mode |= os.O_APPEND
                else:
                    open_mode |= os.O_TRUNC
                log_fileno = os.open(self._log_file, open_mode, 0o644)
                close_log_file = True

            def write_to_file(cnt):
                os.write(log_fileno, cnt)
                os.fsync(log_fileno)  # to avoid buffering on network drive

        else:
            log_fileno = None
            close_log_file = False
            write_to_file = None

        def on_output(cnt):
            if write_to_stdout:
                write_to_stdout(cnt)
            if write_to_file:
                write_to_file(cnt)
            if parse_log:
                parse_log(cnt)

        # run the program
        try:
            with exec_proc(args=args,
                           on_stdout=on_output,
                           stderr_to_stdout=True,
                           env=env,
                           cwd=self._work_dir) as proc:
                yield proc
        finally:
            if close_log_file:
                os.close(log_fileno)
            if self._log_parser:
                try:
                    self._log_parser.flush()
                except Exception as ex:
                    getLogger(__name__).warning(
                        'Error in parsing output of: %s', args, exc_info=True)

    def run(self):
        """Run the program, and get the exit code."""
        with self.exec_proc() as proc:
            return proc.wait()


class ExperimentDoc(object):
    """
    Class to store the experiment document, and to push updates of the
    document to the server in background thread.
    """

    KEYS_TO_EXPAND = ('result', 'webui')

    def __init__(self,
                 client: MLStorageClient,
                 value: DocumentType,
                 heartbeat_interval: int = 120):
        """
        Construct a new :class:`ExperimentDoc`.

        Args:
            client: The MLStorage client.
            value: The initial experiment document.
            heartbeat_interval: The interval (seconds) between to heartbeats.
        """
        self._client = client
        self._value = value
        self._interval = heartbeat_interval
        self._updates = None  # type: DocumentType

        # state of the background worker
        self._stopped = False
        self._thread = None  # type: Thread
        self._heartbeat_time = 0.
        self._update_lock = RLock()
        self._wait_cond = Condition()

    @property
    def client(self) -> MLStorageClient:
        """Get the MLStorage client."""
        return self._client

    @property
    def value(self) -> DocumentType:
        """Get the experiment document."""
        return self._value

    @property
    def id(self) -> IdType:
        """Get the experiment id."""
        return self._value['id']

    @property
    def updates(self) -> Optional[DocumentType]:
        """Get the pending updates."""
        return self._updates

    def merge_doc_updates(self) -> DocumentType:
        """Merge the pending updates into the document locally."""
        with self._update_lock:
            doc = copy.deepcopy(self._value)
            updates = self._updates
            if updates:
                for key, val in updates.items():
                    segments = key.split('.', 1)
                    if len(segments) > 1 and segments[0] in self.KEYS_TO_EXPAND:
                        if segments[0] not in doc:
                            doc[segments[0]] = {}
                        doc[segments[0]][segments[1]] = val
                    else:
                        doc[key] = val
            return doc

    def update(self, fields: DocumentType):
        """
        Update the experiment document.

        This method will queue the updates in background thread.

        Args:
            fields: The new fields.
        """
        with self._update_lock:
            updates = self._updates
            if updates is None:
                updates = {}

            for key, value in fields.items():
                if key in self.KEYS_TO_EXPAND:
                    # special treatment: flatten the result dict
                    if value:
                        for result_key, result_val in value.items():
                            updates[f'{key}.{result_key}'] = result_val
                else:
                    updates[key] = value

            self._updates = updates

            if not self._stopped:
                with self._wait_cond:
                    self._wait_cond.notify_all()

    def set_finished(self,
                     status: str,
                     updates: Optional[DocumentType] = None,
                     retry_intervals: Sequence[int] = (10, 20, 30, 50, 80,
                                                       130, 210)):
        """
        Set the experiment to be finished.

        Args:
            status: The finish status, one of ``{"COMPLETED", "FAILED"}``.
            retry_intervals: The intervals to sleep between two attempts
                to save the finish status.
            updates: The other fields to be updated.
        """
        # gather all update fields
        with self._update_lock:
            if updates:
                self.update(updates)
            updates = self._updates
            self._updates = None

        # try to save the final status
        last_ex = None

        for itv in (0,) + tuple(retry_intervals):
            if itv > 0:
                time.sleep(itv)

            try:
                self._value = self.client.set_finished(self.id, status, updates)
            except Exception as ex:
                last_ex = ex
                getLogger(__name__).warning(
                    'Failed to store the final status of the experiment %s',
                    self.id, exc_info=True
                )
            else:
                last_ex = None
                break

        if last_ex is not None:
            raise last_ex

    def _background_worker(self):
        while not self._stopped:
            # check whether or not there is new updates
            with self._update_lock:
                if self._updates is not None:
                    updates = self._updates
                    self._updates = None
                else:
                    updates = None

            # save the updates
            try:
                self._value = self.client.update(self.id, updates)
            except Exception:
                # failed to save the updates, so we re-queue the updates
                with self._update_lock:
                    if self._updates is None:
                        self._updates = updates
                    else:
                        for key, val in updates.items():
                            if key not in self._updates:
                                self._updates[key] = val

                getLogger(__name__).warning(
                    'Failed to save the document of experiment %s',
                    self.id, exc_info=True
                )

            # set heartbeat
            elapsed = time.time() - self._heartbeat_time
            if elapsed > self._interval:
                try:
                    self.client.heartbeat(self.id)
                except Exception:
                    getLogger(__name__).warning(
                        'Failed to send heartbeat.', exc_info=True)
                finally:
                    # whether or not the heartbeat succeeded, we should
                    # record the time that we sent the heartbeat.
                    now_time = time.time()
                    self._heartbeat_time = now_time
                    elapsed = now_time - self._heartbeat_time

            # wait for the next heartbeat time, or new updates arrived
            with self._wait_cond:
                if not self._stopped:
                    sleep_itv = max(0., self._interval - elapsed)
                    if sleep_itv > 0:
                        self._wait_cond.wait(sleep_itv)

    def start_worker(self):
        """Start the background thread."""
        if self._thread is not None:  # pragma: no cover
            raise RuntimeError('Background thread has already started.')

        self._stopped = False
        self._thread = Thread(target=self._background_worker, daemon=True)
        self._thread.start()

    def stop_worker(self):
        """Stop the background thread."""
        if self._thread is not None:
            with self._wait_cond:
                self._stopped = True
                self._wait_cond.notify_all()
            self._thread.join()
            self._thread = None


class SourceCopier(object):
    """Class to clone source files to destination directory."""

    def __init__(self,
                 source_dir: str,
                 dest_dir: str,
                 includes: List[PatternType],
                 excludes: List[PatternType]):
        """
        Construct a new :class:`SourceCopier`.

        Args:
            source_dir: The source directory.
            dest_dir: The destination directory.
            includes: Path patterns to include.
            excludes: Path patterns to exclude.
        """
        self._source_dir = source_dir
        self._dest_dir = dest_dir
        self._includes = includes
        self._excludes = excludes
        self._created_dirs = []
        self._copied_files = []

    def _walk(self, dir_callback, file_callback):
        def walk(src, relpath):
            dir_callback(src, relpath)

            for name in os.listdir(src):
                src_path = os.path.join(src, name)
                dst_path = f'{relpath}/{name}' if relpath else name

                is_included = lambda: \
                    any(p.match(src_path) for p in self._includes)
                is_not_excluded = lambda: \
                    all(not p.match(src_path) for p in self._excludes)

                if os.path.isdir(src_path):
                    if is_not_excluded():
                        walk(src_path, dst_path)
                else:
                    if is_included() and is_not_excluded():
                        file_callback(src_path, dst_path)

        walk(self._source_dir, '')

    def pack_zip(self, zip_path):
        """
        Pack the source files as a zip file, at `zip_path`.

        Args:
            zip_path: Path of the zip file.
        """
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED
                             ) as zip_file:
            def _create_dir(src, relpath):
                if relpath:
                    zip_file.write(src, arcname=relpath)

            def _copy_file(src, relpath):
                zip_file.write(src, arcname=relpath)

            self._walk(_create_dir, _copy_file)

    def clone_dir(self):
        """
        Copy the source files from `source_dir` to `dest_dir`.
        """
        def _create_dir(src, relpath):
            path = os.path.join(self._dest_dir, relpath)
            self._created_dirs.append(path)
            os.makedirs(path, exist_ok=True)

        def _copy_file(src, relpath):
            path = os.path.join(self._dest_dir, relpath)
            self._copied_files.append(path)
            shutil.copyfile(src, path)

        self._walk(_create_dir, _copy_file)

    def cleanup_dir(self):
        """
        Clean-up the source files at `dest_dir`, which are copied by
        :meth:`clone_dir()`.
        """
        for copied_file in reversed(self._copied_files):
            try:
                if os.path.isfile(copied_file):
                    os.remove(copied_file)
            except Exception:  # pragma: no cover
                getLogger(__name__).warning(
                    'Failed to delete cloned source file: %s',
                    copied_file, exc_info=True
                )

        for created_dir in reversed(self._created_dirs):
            try:
                if os.path.isdir(created_dir) and \
                        len(os.listdir(created_dir)) == 0:
                    os.rmdir(created_dir)
            except Exception:  # pragma: no cover
                getLogger(__name__).warning(
                    'Failed to delete created source directory: %s',
                    created_dir, exc_info=True
                )

    def __enter__(self):
        self.clone_dir()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_dir()


class JsonFileWatcher(object):

    def __init__(self, root_dir: str, file_names: Iterable[str],
                 interval: int = 120):
        self._root_dir = root_dir
        self._file_names = tuple(file_names)
        self._last_check = {}

        self._events = EventHost()
        self._on_json_updated = self.events['json_updated']

        self._thread = None  # type: Thread
        self._stopped = False
        self._wait_cond = Condition()
        self._interval = interval

    @property
    def root_dir(self) -> str:
        return self._root_dir

    @property
    def file_names(self) -> Tuple[str, ...]:
        return self._file_names

    @property
    def events(self) -> EventHost:
        return self._events

    @property
    def on_json_updated(self) -> Event:
        """
        The event that a JSON file has been updated.

        Callback function type: `(file_name: str, json_content: dict) -> None`
        """
        return self._on_json_updated

    def check_files(self, force: bool = True):
        """
        Check the content of JSON files.

        Args:
            force: Whether or not to force load the JSON files, even
                if the mtime and file size has not changed.
        """
        # check the files
        for name in self.file_names:
            path = os.path.join(self.root_dir, name)

            try:
                need_check = False
                st = os.stat(path)

                if not stat.S_ISDIR(st.st_mode):
                    if force:
                        need_check = True
                    else:
                        last_size, last_mtime = \
                            self._last_check.get(name, (None, None))
                        need_check = (last_size != st.st_size or
                                      last_mtime != st.st_mtime)

                if need_check:
                    with codecs.open(path, 'rb', 'utf-8') as f:
                        content = json_loads(f.read())
                    getLogger(__name__).debug('JSON content updated: %s', path)
                    self.on_json_updated.fire(name, content)
                    self._last_check[name] = (st.st_size, st.st_mtime)

            except IOError:
                getLogger(__name__).debug(
                    'IO error when checking the JSON file: %s',
                    path, exc_info=True
                )
            except Exception as ex:
                getLogger(__name__).warning(
                    'Failed to store the content from JSON file: %s',
                    path, exc_info=True
                )

    def _background_worker(self):
        while not self._stopped:
            self.check_files(force=False)
            with self._wait_cond:
                if not self._stopped:
                    self._wait_cond.wait(self._interval)

    def start_worker(self):
        """Start the background thread."""
        if self._thread is not None:  # pragma: no cover
            raise RuntimeError('Background thread has already started.')

        self._stopped = False
        self._thread = Thread(target=self._background_worker, daemon=True)
        self._thread.start()

    def stop_worker(self):
        """Stop the background thread."""
        if self._thread is not None:
            with self._wait_cond:
                self._stopped = True
                self._wait_cond.notify_all()
            self._thread.join()
            self._thread = None

    def __enter__(self):
        self.start_worker()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.stop_worker()
        finally:
            self.check_files(force=True)

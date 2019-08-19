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
from http.server import BaseHTTPRequestHandler, HTTPServer
from logging import getLogger, FileHandler
from socketserver import ThreadingMixIn
from threading import RLock, Condition, Thread
from typing import *
from urllib.parse import urlsplit, urlunsplit, SplitResult

import click

from .config import Config, ConfigLoader, deep_copy
from .events import EventHost, Event
from .mlstorage import (DocumentType, MLStorageClient, IdType,
                        normalize_relpath)
from .utils import exec_proc, timed_wait_proc, json_loads

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
    parent_id: str = None
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
        src_dir: str = '.'
        dst_dir: str = ''
        includes = [
            re.compile(r'.*\.(py|pl|rb|js|sh|r|bat|cmd|exe|jar)$')
        ]
        excludes = [
            re.compile(
                r'.*[\\/](node_modules|\.svn|\.cvs|\.idea|'
                r'\.DS_Store|\.git|\.hg|\.pytest_cache|__pycache__)'
                r'(?:$|[\\/].*)'
            )
        ]

        copy_to_dst: bool = False
        cleanup: bool = True

        make_archive: bool = True
        archive_name: str = 'source.zip'

        def user_validate(self):
            def maybe_compile(p):
                if not hasattr(p, 'match'):
                    p = re.compile(p)
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

        class stdout_pattern(Config):
            mltk: Optional[str] = None
            mltk_metric: Optional[str] = None

        config_file: str = 'config.json'
        default_config_file: str = 'config.defaults.json'
        result_file: str = 'result.json'
        webui_file: str = 'webui.json'

    class logging(Config):
        log_file: str = 'console.log'
        daemon_log_file: str = 'daemon.log'
        runner_log_file: str = 'mlrun.log'

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
    """The machine learning experiment runner."""

    def __init__(self,
                 config: MLRunnerConfig,
                 retry_intervals: Sequence[float] = (10, 20, 30, 50, 80,
                                                     130, 210)):
        """
        Construct a new :class:`MLRunner`.

        Args:
            config: The runner configuration.
            retry_intervals: The intervals to sleep between two attempts
                to save the finish status.
        """
        self._config = config
        self._client = MLStorageClient(config.server)
        self._retry_intervals = tuple(retry_intervals)
        self._doc = None  # type: ExperimentDoc
        self._clone_doc = None  # type: DocumentType

    @property
    def config(self) -> MLRunnerConfig:
        """Get the runner configuration."""
        return self._config

    @property
    def client(self) -> MLStorageClient:
        """Get the MLStorage client."""
        return self._client

    @property
    def retry_intervals(self) -> Tuple[float]:
        return self._retry_intervals

    @property
    def doc(self) -> 'ExperimentDoc':
        """Get the experiment document object."""
        return self._doc

    def env_dict(self, experiment_id, output_dir, work_dir) -> Dict[str, str]:
        """
        Get the environment dict for this process.

        Args:
            experiment_id: The experiment id.
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
            'MLSTORAGE_SERVER_URI': self.config.server,
            'MLSTORAGE_EXPERIMENT_ID': str(experiment_id),
            'MLSTORAGE_OUTPUT_DIR': output_dir,
        })
        if self.config.env is not None:
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
        except IOError:  # pragma: no cover
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
                if v[1] is not None:
                    return f'{v[0]} (±{v[1]})'
                else:
                    return v[0]

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
            def get_hostname_cached():
                if _cache[0] is None:
                    _cache[0] = socket.gethostname()
                return _cache[0]
            _cache = [None]  # type: List[str]

            webui = {}
            for key, val in info.items():
                webui[key] = val
                try:
                    u = urlsplit(val)
                except Exception as ex:  # pragma: no cover
                    pass
                else:
                    if u.scheme in ('http', 'https'):
                        parts = u.netloc.rsplit(':', 1)
                        if parts[0] in ('0.0.0.0', '[::0]'):
                            parts[0] = get_hostname_cached()
                            u = SplitResult(
                                scheme=u.scheme, netloc=':'.join(parts),
                                path=u.path, query=u.query, fragment=u.fragment
                            )
                            webui[key] = urlunsplit(u)

            self.doc.update({'webui': webui})

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
        env = self.env_dict(
            experiment_id=self.doc.id,
            output_dir=output_dir,
            work_dir=work_dir
        )

        # update the doc with execution info
        self.doc.update({
            'args': self.config.args,
            'exc_info': {
                'hostname': socket.gethostname(),
                'env': env,
            }
        })

        # create the program host for the main process
        log_file = os.path.join(output_dir, self.config.logging.log_file)
        main_host = ProgramHost(
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
                a, b = os.path.splitext(self.config.logging.daemon_log_file)
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
            source_dir=self.config.source.src_dir,
            dest_dir=os.path.join(output_dir, self.config.source.dst_dir),
            includes=self.config.source.includes,
            excludes=self.config.source.excludes,
            cleanup=self.config.source.cleanup,
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

        # initialize the control server
        control_server = ControlServer()
        control_server.on_kill.do(main_host.kill)

        # now execute the processes
        @contextmanager
        def wrap_daemon_process(daemon):
            proc = None
            try:
                with daemon.exec_proc() as p:
                    proc = p
                    yield proc
            finally:
                if proc is not None:
                    code = proc.poll()
                    if code is not None:
                        getLogger(__name__).info(f'Daemon process {proc.pid} '
                                                 f'exited with code: {code}')

        with ExitStack() as ctx_stack:
            # populate the work directory with source files
            source_copier.copy_args_files(self.config.args)

            if self.config.source.copy_to_dst:
                ctx_stack.enter_context(source_copier)
                getLogger(__name__).info(
                    f'Cloned {source_copier.file_count} source files to '
                    f'the output directory.'
                )

            if self.config.source.make_archive:
                archive_name = os.path.join(
                    output_dir, self.config.source.archive_name)
                source_copier.pack_zip(archive_name)
                getLogger(__name__).info(
                    'Created source file archive: %s', archive_name)

            # add a temporary file cleaner
            ctx_stack.enter_context(TemporaryFileCleaner(output_dir))
            getLogger(__name__).debug('Temporary file cleaner initialized.')

            # start the background json watcher
            ctx_stack.enter_context(json_watcher)
            getLogger(__name__).debug('JSON file watcher started.')

            # start the daemon processes
            if daemons and self.config.daemon:
                for daemon, daemon_args in zip(daemons, self.config.daemon):
                    daemon_proc = ctx_stack.enter_context(
                        wrap_daemon_process(daemon))
                    getLogger(__name__).info(
                        f'Started daemon process {daemon_proc.pid}: %s',
                        daemon_args
                    )

            # run the main process
            with main_host.exec_proc() as proc, \
                    control_server.run_in_background():
                getLogger(__name__).info(
                    'Started experiment process %s: %s',
                    proc.pid, self.config.args
                )
                getLogger(__name__).info(
                    'Control server started at: %s', control_server.uri)

                # update the doc with execution info
                self.doc.update({
                    'exc_info': {
                        'pid': proc.pid,
                    },
                    'control_port': {
                        'kill': control_server.uri + '/kill',
                    }
                })

                # wait for the process to exit
                _ = proc.wait()

            code = proc.poll()
            getLogger(__name__).info(
                f'Experiment process exited with code: {code}')

            return code

    def run(self):
        """Run the experiment."""
        # get the previous experiment if we are cloning from any one
        if self.config.clone_from:
            self._clone_doc = self.client.get(self.config.clone_from)

        # create an experiment, or resume from the previous experiment
        meta = {}
        for key in ('name', 'description', 'tags'):
            if self.config[key] is not None:
                meta[key] = self.config[key]

        parent_id = self.config.parent_id
        if parent_id is None:
            parent_id = os.environ.get('MLSTORAGE_EXPERIMENT_ID', None)
        if parent_id is not None:
            meta['parent_id'] = parent_id

        if 'name' not in meta:
            # name is required on some version, so if it is not specified,
            # generate one
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
        doc_id = doc['_id']

        try:
            output_dir = doc['storage_dir']
            os.makedirs(output_dir, exist_ok=True)

            # function to help filter out None dict items
            def filter_dict(d):
                return {k: v for k, v in d.items() if v is not None}

            # prepare for the work dir
            runner_log_file = os.path.join(
                output_dir, self.config.logging.runner_log_file)

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
                    getLogger(__name__).error(
                        'Failed to run the experiment.', exc_info=True)
                    self.doc.stop_worker()
                    self.doc.set_finished(
                        final_status,
                        filter_dict({
                            'error': {
                                'message': str(ex),
                                'traceback': ''.join(
                                    traceback.format_exception(*sys.exc_info()))
                            },
                            'exit_code': exit_code,
                            'storage_size': fs_size,
                            'storage_inode': inode_count,
                        }),
                        retry_intervals=self.retry_intervals
                    )
                    raise

                else:
                    final_status = 'COMPLETED'
                    self.doc.stop_worker()
                    self.doc.set_finished(
                        final_status,
                        filter_dict({
                            'exit_code': exit_code,
                            'storage_size': fs_size,
                            'storage_inode': inode_count,
                        }),
                        retry_intervals=self.retry_intervals
                    )

        except Exception as ex:
            if self.doc is None or not self.doc.has_set_finished:
                getLogger(__name__).error(
                    'Failed to run the experiment.', exc_info=True)
                self.client.set_finished(doc_id, 'FAILED', {
                    'error': {
                        'message': str(ex),
                        'traceback': ''.join(
                            traceback.format_exception(*sys.exc_info()))
                    }
                })
            raise

        return exit_code


@click.command()
@click.option('-C', '--config-file', required=False, multiple=True,
              help='Load runner configuration from JSON or YAML file. '
                   'Values from all config files will be merged. '
                   'The CLI arguments will override all config files.')
@click.option('-n', '--name', required=False, default=None,
              help='Experiment name.')
@click.option('-d', '--description', required=False, default=None,
              help='Experiment description.')
@click.option('-t', '--tags', required=False, multiple=True, default=None,
              help='Experiment tags, comma separated strings, e.g. '
                   '"prec 0.996, state of the arts".')
@click.option('-e', '--env', required=False, multiple=True,
              help='Environmental variable (FOO=BAR).')
@click.option('-g', '--gpu', required=False, multiple=True,
              help='Quick approach to set the "CUDA_VISIBLE_DEVICES" '
                   'environmental variable.')
@click.option('-w', '--work-dir', required=False, default=None,
              help='Use this work directory, instead of using the MLStorage '
                   'output dir.')
@click.option('-s', '--server', required=False, default=None,
              help='Specify the URI of MLStorage API server, e.g., '
                   '"http://localhost:8080".  If not specified, will use '
                   '``os.environ["MLSTORAGE_SERVER_URI"]``.')
@click.option('--resume-from', required=False, default=None,
              help='ID of the experiment to resume from.')
@click.option('--clone-from', required=False, default=None,
              help='ID of the experiment to clone from.')
@click.option('--copy-source/--no-copy-source',  'copy_source',
              is_flag=True, default=None, required=False,
              help='Whether or not to copy the source files from current '
                   'directory to MLStorage output directory?')
@click.option('--source-archive/--no-source-archive', 'source_archive',
              is_flag=True, default=None, required=False,
              help='Whether or not to pack the source files from the current '
                   'directory into a zip archive in the MLStorage output '
                   'directory?')
@click.option('--parse-stdout/--no-parse-stdout', 'parse_stdout',
              is_flag=True, required=False, default=None,
              help='Whether or not to parse the output of experiment and '
                   'daemon processes?')
@click.option('-D', '--daemon', required=False, multiple=True,
              help='Specify the shell command of daemon processes, to be '
                   'executed along with the main experiment process.')
@click.option('--tensorboard', is_flag=True, required=False, default=None,
              help='Launch a TensorBoard server in the work directory. '
                   'This is equivalent to '
                   '`-D "tensorboard --logdir=. --port=0"`')
@click.option('-c', 'command', required=False, default=None,
              help='Specify the shell command to execute. '
                   'Will override the program arguments (args).')
@click.argument('args', nargs=-1)
def mlrun(config_file, name, description, tags, env, gpu, work_dir, server,
          resume_from, clone_from, copy_source, source_archive, parse_stdout,
          daemon, tensorboard, command, args):
    """
    Run an experiment.

    The program arguments should be either specified at the end, after a "--"
    mark, or specified as command line via "-c" argument.  For example::

        mlrun -- python train.py
        mlrun -c "python train.py"

    By default, the program will not run in the current directory.
    Instead, it will run in the MLStorage output directory, assigned by the
    server.  Source files appear in the program command line or arguments
    will be copied to the output directory.  If you need to copy other source
    files to the output directory, you may specify the "--copy-source" argument,
    for example::

        mlrun --copy-source -- python train.py

    The following file extensions will be regarded as source files::

        *.py *.pl *.rb *.js *.sh *.r *.bat *.cmd *.exe *.jar

    The source files will be collected into "<MLStorage output dir>/source.zip",
    unless you specify "--no-source-archive".

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
        |-- ... (copied source files)
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

    # parse the daemon
    if tensorboard:
        daemon = list(daemon or [])
        daemon.append('tensorboard --logdir=. --port=0')

    # feed CLI arguments into MLRunnerConfig
    cli_config = {
        'name': name,
        'description': description,
        'tags': tags or None,
        'env': env_dict,
        'gpu': gpu_list,
        'work_dir': work_dir,
        'server': server or os.environ.get('MLSTORAGE_SERVER_URI', None),
        'resume_from': resume_from,
        'clone_from': clone_from,
        'source.copy_to_dst': copy_source,
        'source.make_archive': source_archive,
        'integration.parse_stdout': parse_stdout,
        'daemon': daemon or None,
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
    {'SimpleHTTP': 'http://0.0.0.0:8000/'}

    >>> parser.parse_line(b'no pattern exist')

    Too long output lines will not be parsed, for example:

    >>> parser.parse_line(b'[Epoch 1/10, Step 5] ')
    {'epoch': 1, 'max_epoch': 10, 'step': 5} {}
    >>> parser.parse_line(b'[Epoch 1/10, Step 5]' + b' ' * 2048)
    """

    def __init__(self,
                 max_parse_length: int = 2048,
                 mltk_pattern: Optional[str] = None,
                 mltk_metric_pattern: Optional[str] = None):
        """
        Construct a new :class:`StdoutParser`.

        Args:
            max_parse_length: The maximum length of line to parse.
            mltk_pattern: The regex pattern for the overall MLTK logs.
            mltk_metric_pattern: The regex pattern for the metrics in MLTK logs.
        """
        self._max_parse_length = max_parse_length
        self._events = EventHost()
        self._on_mltk_log = self.events['on_mltk_log']
        self._on_webui_log = self.events['on_webui_log']

        # buffer for accumulating lines
        self._line_buffer = None

        # patterns for parsing tfsnippet & mltk logs
        if mltk_pattern is None:
            mltk_pattern = (
                rb'^\['
                rb'(?:Epoch (?P<epoch>\d+)(?:/(?P<max_epoch>\d+))?)?[, ]*'
                rb'(?:Step (?P<step>\d+)(?:/(?P<max_step>\d+))?)?[, ]*'
                rb'(?:ETA (?P<eta>[0-9\.e+ dhms]+))?'
                rb'\]\s*'
                rb'(?P<metrics>.*?)\s*(?:\(\*\))?\s*'
                rb'$'
            )
        else:
            mltk_pattern = str(mltk_pattern).encode('utf-8')

        if mltk_metric_pattern is None:
            mltk_metric_pattern = (
                rb'^\s*(?P<name>[^:]+): (?P<mean>[^()]+)'
                rb'(?: \(\xc2\xb1(?P<std>[^()]+)\))?\s*$'
            )
        else:
            mltk_metric_pattern = str(mltk_metric_pattern).encode('utf-8')

        self._mltk_pattern = re.compile(mltk_pattern)
        self._mltk_metric_pattern = re.compile(mltk_metric_pattern)

        # pattern for parsing tensorboard log
        self._webui_pattern = re.compile(
            rb'(?:^TensorBoard \S+ at (?P<TensorBoard>\S+))|'
            rb'(?:^Serving HTTP on \S+ port \d+ \((?P<SimpleHTTP>[^()]+)\))'
        )

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
            for key, val in g.items():
                if val is not None:
                    self.on_webui_log.fire({key: val.decode('utf-8')})
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
        self._proc = None  # type: subprocess.Popen

    @property
    def proc(self) -> subprocess.Popen:
        """Get the managed process object."""
        return self._proc

    def kill(self, ctrl_c_timeout: float = 3):
        """
        Kill the process if it is running.

        This method will first try to interrupt the process by Ctrl+C.
        If the process does not exit in `ctrl_c_timeout` seconds, then
        it will kill the process by SIGKILL (or terminate on windows).
        """
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.kill(ctrl_c_timeout=ctrl_c_timeout)
            except ProcessLookupError:  # pragma: no cover
                # which indicates the process has exited
                pass

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
                self._proc = proc
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

    KEYS_TO_EXPAND = ('config', 'result', 'webui', 'exc_info')

    def __init__(self,
                 client: MLStorageClient,
                 value: DocumentType,
                 heartbeat_interval: float = 120):
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
        self._has_set_finished = False

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
        return self._value['_id']

    @property
    def updates(self) -> Optional[DocumentType]:
        """Get the pending updates."""
        return self._updates

    @property
    def has_set_finished(self) -> bool:
        """
        Whether or not :meth:`set_finished()` has been successfully called.
        """
        return self._has_set_finished

    def merge_doc_updates(self) -> DocumentType:
        """Merge the pending updates into the document locally."""
        with self._update_lock:
            doc = deep_copy(self._value)
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

    def update(self, fields: DocumentType, notify_worker: bool = True):
        """
        Update the experiment document.

        This method will queue the updates in background thread.

        Args:
            fields: The new fields.
            notify_worker: Whether or not to notify the background worker?
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

        if notify_worker:
            with self._wait_cond:
                self._wait_cond.notify_all()

    def set_finished(self,
                     status: str,
                     updates: Optional[DocumentType] = None,
                     retry_intervals: Sequence[float] = (10, 20, 30, 50, 80,
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
                self.update(updates, notify_worker=False)
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
                self._has_set_finished = True
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
                if updates:
                    with self._update_lock:
                        if self._updates is None:
                            self._updates = copy.deepcopy(updates)
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

    def __enter__(self):
        self.start_worker()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_worker()


class SourceCopier(object):
    """Class to clone source files to destination directory."""

    def __init__(self,
                 source_dir: str,
                 dest_dir: str,
                 includes: List[PatternType],
                 excludes: List[PatternType],
                 cleanup: bool = False):
        """
        Construct a new :class:`SourceCopier`.

        Args:
            source_dir: The source directory.
            dest_dir: The destination directory.
            includes: Path patterns to include.
            excludes: Path patterns to exclude.
        """
        self._source_dir = os.path.normpath(os.path.abspath(source_dir))
        self._dest_dir = os.path.abspath(dest_dir)
        self._includes = includes
        self._excludes = excludes
        self._created_dirs = []
        self._copied_files = []
        self._cleanup = cleanup

    @property
    def file_count(self) -> int:
        return len(self._copied_files)

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

    def copy_args_files(self, args: Union[str, List[str]]):
        """
        Copy the source files specified in `args` to `dest_dir`.

        Args:
            args: The CLI arguments.

        Notes:
            Files copied by this method will not be deleted in :meth:`cleanup()`
        """
        if isinstance(args, (str, bytes)):
            args = shlex.split(str(args))

        for arg in args:
            arg_path = os.path.normpath(
                os.path.abspath(os.path.join(self._source_dir, arg)))

            if os.path.isfile(arg_path) and \
                    any(p.match(arg_path) for p in self._includes) and \
                    not any(p.match(arg_path) for p in self._excludes):
                try:
                    arg_relpath = normalize_relpath(
                        os.path.relpath(arg_path, self._source_dir))
                except ValueError:
                    pass  # not a file in `source_dir`
                else:
                    dst_path = os.path.join(self._dest_dir, arg_relpath)
                    dst_dir = os.path.dirname(dst_path)
                    if not os.path.isdir(dst_dir):
                        os.makedirs(dst_dir, exist_ok=True)
                    shutil.copyfile(arg_path, dst_path)

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
            if not os.path.exists(path):  # do not override existing file
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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cleanup:
            self.cleanup_dir()


class TemporaryFileCleaner(object):
    """
    Class to cleanup some common temporary files generated by the experiment
    program, e.g., "*.pyc".
    """

    TEMP_FILE_PATTERNS = (
        re.compile(r'.*\.(pyc|)$'),
        re.compile(r'.*[\\/](Thumbs\.db|\.DS_Store)$'),
    )
    TEMP_DIR_PATTERNS = (
        re.compile(r'.*[\\/](__pycache__)(?:$|[\\/].*)'),
    )
    EXCLUDE_PATTERNS = (
        re.compile(
            r'.*[\\/](\.svn|\.cvs|\.idea|\.git|\.hg)'
            r'(?:$|[\\/].*)'
        ),
    )

    def __init__(self,
                 root_dir: str,
                 file_patterns: Iterable[PatternType] = TEMP_FILE_PATTERNS,
                 dir_patterns: Iterable[PatternType] = TEMP_DIR_PATTERNS,
                 exclude_patterns: Iterable[PatternType] = EXCLUDE_PATTERNS):
        """
        Construct a new :class:`TemporaryFileCleaner`.

        Args:
            root_dir: The root directory.
            file_patterns: The temporary file patterns.
            dir_patterns: The temporary directory patterns.  Only empty
                temporary directories will be removed.
            exclude_patterns: The file and directory patterns to skip.
        """
        self._root_dir = os.path.abspath(root_dir)
        self._file_patterns = tuple(file_patterns)
        self._dir_patterns = tuple(dir_patterns)
        self._exclude_patterns = tuple(exclude_patterns)

    @property
    def root_dir(self):
        return self._root_dir

    def _cleanup_dir(self, path):
        for name in os.listdir(path):
            f_path = os.path.join(path, name)

            try:
                if any(p.match(f_path) for p in self._exclude_patterns):
                    # excluded, skip
                    continue

                if os.path.isdir(f_path):
                    # cleanup the directory
                    self._cleanup_dir(f_path)
                    if any(p.match(f_path) for p in self._dir_patterns) and \
                            len(os.listdir(f_path)) == 0:
                        os.rmdir(f_path)  # remove it if matched dir pattern
                else:
                    # cleanup the file
                    if any(p.match(f_path) for p in self._file_patterns):
                        os.remove(f_path)
            except Exception as ex:
                getLogger(__name__).warning('Failed to cleanup: %s', f_path)

    def cleanup(self):
        try:
            if os.path.isdir(self.root_dir):
                self._cleanup_dir(self.root_dir)
        except Exception as ex:
            getLogger(__name__).warning('Failed to cleanup: %s', self.root_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class JsonFileWatcher(object):
    """
    Class to watch the changes of JSON files.
    """

    def __init__(self, root_dir: str, file_names: Iterable[str],
                 interval: float = 120):
        """
        Construct a new :class:`JsonFileWatcher`.

        Args:
            root_dir: Root directory of the files.
            file_names: Names of the files.
            interval: Check interval in seconds.
        """
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
                    self._last_check[name] = (st.st_size, st.st_mtime)
                    self.on_json_updated.fire(name, content)

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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.stop_worker()
        finally:
            self.check_files(force=True)


class ControlServerHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_error(404, 'Not Found')

    def do_POST(self):
        if self.path == '/kill':
            self.handle_kill()
        else:
            self.send_error(404, 'Not Found')

    def handle_kill(self):
        try:
            self.server.on_kill.fire()

            self.send_response(200, 'Ok')
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', 2)
            self.end_headers()
            self.wfile.write(b'{}')
        except Exception as ex:
            getLogger(__name__).warning(
                'Failed to kill the process.', exc_info=True)
            self.send_error(500, 'Failed to kill the process.')


class ControlServer(HTTPServer, ThreadingMixIn):
    """
    A server that exposes control APIs on the experiment process.
    """

    def __init__(self, host='', port=0):
        """
        Construct a new :class:`ControlPortServer`.

        Args:
            host: The host to bind.
            port: The port to bind.
        """
        super(ControlServer, self).__init__(
            (host, port), ControlServerHandler)
        self._host = host
        self._port = port
        self._events = EventHost()
        self._on_kill = self.events['on_kill']

    @property
    def events(self) -> EventHost:
        """Get the event host."""
        return self._events

    @property
    def on_kill(self) -> Event:
        """Get the event that kill request is received."""
        return self._on_kill

    @property
    def uri(self) -> str:
        """Get the URI of the server."""
        host = (self._host if self._host not in ('', '0.0.0.0')
                else socket.gethostname())
        port = self.socket.getsockname()[1]
        return f'http://{host}:{port}'

    @contextmanager
    def run_in_background(self):
        """Run the server in background."""
        th = Thread(target=self.serve_forever)
        try:
            th.daemon = True
            th.start()
            yield self
        finally:
            self.shutdown()
            th.join()

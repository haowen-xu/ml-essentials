import os
import signal
import subprocess
import sys
import types
from contextlib import contextmanager
from logging import getLogger
from threading import Thread
from typing import *

__all__ = ['timed_wait_proc', 'exec_proc']

OutputCallbackType = Callable[[bytes], None]


def timed_wait_proc(proc: subprocess.Popen, timeout: float) -> Optional[int]:
    """
    Wait a process for at most `timeout` seconds.

    Args:
        proc: The process to wait.
        timeout: The timeout seconds.

    Returns:
        The exit code, or :obj:`None` if the process does not exit.
    """
    try:
        return proc.wait(timeout)
    except subprocess.TimeoutExpired:
        return None


def recursive_kill(proc: subprocess.Popen,
                   ctrl_c_timeout: float = 3,
                   kill_timeout: float = 10) -> Optional[int]:
    """
    Recursively kill a process tree.

    Args:
        proc: The process to kill.
        ctrl_c_timeout: Seconds to wait for the program to respond to
            CTRL+C signal.
        kill_timeout: Seconds to wait for the program to be killed.

    Returns:
        The return code, or None if the process cannot be killed.
    """
    try:
        gid = os.getpgid(proc.pid)
    except Exception:
        # indicate pid does not exist
        return

    # try to kill the process by ctrl+c
    ctrl_c_event = (signal.SIGINT if sys.platform != 'win32'
                    else signal.CTRL_C_EVENT)
    os.killpg(gid, ctrl_c_event)
    code = timed_wait_proc(proc, ctrl_c_timeout)
    if code is None:
        print(f'Failed to kill sub-process {proc.pid} by SIGINT, plan to kill '
              f'it by SIGTERM.')
    else:
        return code

    # try to kill the process by SIGTERM
    os.killpg(gid, signal.SIGTERM)
    code = timed_wait_proc(proc, kill_timeout)
    if code is None:
        print(f'Failed to kill sub-process {proc.pid} by SIGTERM, plan to kill '
              f'it by SIGKILL.')
    else:
        return code

    # try to kill the process by SIGKILL
    os.killpg(gid, signal.SIGKILL)
    code = timed_wait_proc(proc, kill_timeout)
    if code is None:
        print(f'Failed to kill sub-process {proc.pid} by SIGKILL, give up.')

    return code


@contextmanager
def exec_proc(args: Union[str, Iterable[str]],
              on_stdout: OutputCallbackType = None,
              on_stderr: OutputCallbackType = None,
              stderr_to_stdout: bool = False,
              buffer_size: int = 16 * 1024,
              ctrl_c_timeout: float = 3,
              **kwargs) -> Generator[subprocess.Popen, None, None]:
    """
    Execute an external program within a context.

    Args:
        args: Command line or arguments of the program.
            If it is a command line, then `shell = True` will be set.
        on_stdout: Callback for capturing stdout.
        on_stderr: Callback for capturing stderr.
        stderr_to_stdout: Whether or not to redirect stderr to stdout?
            If specified, `on_stderr` will be ignored.
        buffer_size: Size of buffers for reading from stdout and stderr.
        ctrl_c_timeout: Seconds to wait for the program to respond to
            CTRL+C signal.
        \\**kwargs: Other named arguments passed to :func:`subprocess.Popen`.

    Yields:
        The process object.
    """
    # check the arguments
    if stderr_to_stdout:
        kwargs['stderr'] = subprocess.STDOUT
        on_stderr = None
    if on_stdout is not None:
        kwargs['stdout'] = subprocess.PIPE
    if on_stderr is not None:
        kwargs['stderr'] = subprocess.PIPE

    # output reader
    def reader_func(fd, action):
        while not stopped[0]:
            buf = os.read(fd, buffer_size)
            if not buf:
                break
            action(buf)

    def make_reader_thread(fd, action):
        th = Thread(target=reader_func, args=(fd, action))
        th.daemon = True
        th.start()
        return th

    # internal flags
    stopped = [False]

    # launch the process
    stdout_thread = None  # type: Thread
    stderr_thread = None  # type: Thread
    if isinstance(args, (str, bytes)):
        shell = True
    else:
        args = tuple(args)
        shell = False

    if sys.platform != 'win32':
        kwargs.setdefault('preexec_fn', os.setsid)
    proc = subprocess.Popen(args, shell=shell, **kwargs)

    # patch the kill() to ensure the whole process group would be killed,
    # in case `shell = True`.
    def my_kill(self, ctrl_c_timeout=ctrl_c_timeout):
        recursive_kill(self, ctrl_c_timeout=ctrl_c_timeout)

    proc.kill = types.MethodType(my_kill, proc)

    try:
        if on_stdout is not None:
            stdout_thread = make_reader_thread(proc.stdout.fileno(), on_stdout)
        if on_stderr is not None:
            stderr_thread = make_reader_thread(proc.stderr.fileno(), on_stderr)

        try:
            yield proc
        except KeyboardInterrupt:  # pragma: no cover
            if proc.poll() is None:
                # Wait for a while to ensure the program has properly dealt
                # with the interruption signal.  This will help to capture
                # the final output of the program.
                # TODO: use signal.signal instead for better treatment
                _ = timed_wait_proc(proc, 1)

    finally:
        if proc.poll() is None:
            proc.kill()

        # Wait for the reader threads to exit
        stopped[0] = True
        for th in (stdout_thread, stderr_thread):
            if th is not None:
                th.join()

        # Ensure all the pipes are closed.
        for f in (proc.stdout, proc.stderr, proc.stdin):
            if f is not None:
                try:
                    f.close()
                except Exception:  # pragma: no cover
                    getLogger(__name__).info(
                        'Failed to close a sub-process pipe.',
                        exc_info=True
                    )

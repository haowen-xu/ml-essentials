import os
import signal
import subprocess
import sys
from contextlib import contextmanager
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


@contextmanager
def exec_proc(args: Union[str, Iterable[str]],
              on_stdout: OutputCallbackType = None,
              on_stderr: OutputCallbackType = None,
              stderr_to_stdout: bool = False,
              buffer_size: int = 16 * 1024,
              ctrl_c_timeout: int = 3,
              kill_timeout: int = 60,
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
        kill_timeout: Seconds to wait for the program to terminate after
            being killed.
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
        while not giveup_waiting[0]:
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
    giveup_waiting = [False]

    # launch the process
    stdout_thread = None  # type: Thread
    stderr_thread = None  # type: Thread
    if isinstance(args, (str, bytes)):
        shell = True
    else:
        args = tuple(args)
        shell = False
    proc = subprocess.Popen(args, shell=shell, **kwargs)

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
            # First, try to interrupt the process with Ctrl+C signal
            ctrl_c_signal = (signal.SIGINT if sys.platform != 'win32'
                             else signal.CTRL_C_EVENT)
            os.kill(proc.pid, ctrl_c_signal)
            if timed_wait_proc(proc, ctrl_c_timeout) is None:
                # If the Ctrl+C signal does not work, terminate it.
                proc.kill()
            # Finally, wait for at most 60 seconds
            if timed_wait_proc(proc, kill_timeout) is None:  # pragma: no cover
                giveup_waiting[0] = True

        # Close the pipes such that the reader threads will ensure to exit,
        # if we decide to give up waiting.
        def close_pipes():
            for f in (proc.stdout, proc.stderr, proc.stdin):
                if f is not None:
                    f.close()

        if giveup_waiting[0]:  # pragma: no cover
            close_pipes()

        # Wait for the reader threads to exit
        for th in (stdout_thread, stderr_thread):
            if th is not None:
                th.join()

        # Ensure all the pipes are closed.
        if not giveup_waiting[0]:
            close_pipes()



# Copyright 2016 Vijayaditya Peddinti.
#           2016 Vimal Manohar
#           2017 Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

""" This module contains several utility functions and classes that are
commonly used in many kaldi python scripts.
"""

import argparse
import logging
import math
import os
import subprocess
import sys
import threading

try:
    import thread as thread_module
except:
    import _thread as thread_module

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def send_mail(message, subject, email_id):
    try:
        subprocess.Popen(
            'echo "{message}" | mail -s "{subject}" {email}'.format(
                message=message,
                subject=subject,
                email=email_id), shell=True)
    except Exception as e:
        logger.info("Unable to send mail due to error:\n {error}".format(
                        error=str(e)))
        pass


def str_to_bool(value):
    if value == "true":
        return True
    elif value == "false":
        return False
    else:
        raise ValueError


class StrToBoolAction(argparse.Action):
    """ A custom action to convert bools from shell format i.e., true/false
        to python format i.e., True/False """

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, str_to_bool(values))
        except ValueError:
            raise Exception(
                "Unknown value {0} for --{1}".format(values, self.dest))


class NullstrToNoneAction(argparse.Action):
    """ A custom action to convert empty strings passed by shell to None in
    python. This is necessary as shell scripts print null strings when a
    variable is not specified. We could use the more apt None in python. """

    def __call__(self, parser, namespace, values, option_string=None):
        if values.strip() == "":
            setattr(namespace, self.dest, None)
        else:
            setattr(namespace, self.dest, values)


class smart_open(object):
    """
    This class is designed to be used with the "with" construct in python
    to open files. It is similar to the python open() function, but
    treats the input "-" specially to return either sys.stdout or sys.stdin
    depending on whether the mode is "w" or "r".

    e.g.: with smart_open(filename, 'w') as fh:
            print ("foo", file=fh)
    """
    def __init__(self, filename, mode="r"):
        self.filename = filename
        self.mode = mode
        assert self.mode == "w" or self.mode == "r"

    def __enter__(self):
        if self.filename == "-" and self.mode == "w":
            self.file_handle = sys.stdout
        elif self.filename == "-" and self.mode == "r":
            self.file_handle = sys.stdin
        else:
            self.file_handle = open(self.filename, self.mode)
        return self.file_handle

    def __exit__(self, *args):
        if self.filename != "-":
            self.file_handle.close()


class smart_open(object):
    """
    This class is designed to be used with the "with" construct in python
    to open files. It is similar to the python open() function, but
    treats the input "-" specially to return either sys.stdout or sys.stdin
    depending on whether the mode is "w" or "r".

    e.g.: with smart_open(filename, 'w') as fh:
            print ("foo", file=fh)
    """
    def __init__(self, filename, mode="r"):
        self.filename = filename
        self.mode = mode
        assert self.mode == "w" or self.mode == "r"

    def __enter__(self):
        if self.filename == "-" and self.mode == "w":
            self.file_handle = sys.stdout
        elif self.filename == "-" and self.mode == "r":
            self.file_handle = sys.stdin
        else:
            self.file_handle = open(self.filename, self.mode)
        return self.file_handle

    def __exit__(self, *args):
        if self.filename != "-":
            self.file_handle.close()


def check_if_cuda_compiled():
    p = subprocess.Popen("cuda-compiled")
    p.communicate()
    if p.returncode == 1:
        return False
    else:
        return True


def execute_command(command):
    """ Runs a kaldi job in the foreground and waits for it to complete; raises an
        exception if its return status is nonzero.  The command is executed in
        'shell' mode so 'command' can involve things like pipes.  Often,
        'command' will start with 'run.pl' or 'queue.pl'.  The stdout and stderr
        are merged with the calling process's stdout and stderr so they will
        appear on the screen.

        See also: get_command_stdout, background_command
    """
    p = subprocess.Popen(command, shell=True)
    p.communicate()
    if p.returncode is not 0:
        raise Exception("Command exited with status {0}: {1}".format(
                p.returncode, command))


def get_command_stdout(command, require_zero_status = True):
    """ Executes a command and returns its stdout output as a string.  The
        command is executed with shell=True, so it may contain pipes and
        other shell constructs.

        If require_zero_stats is True, this function will raise an exception if
        the command has nonzero exit status.  If False, it just prints a warning
        if the exit status is nonzero.

        See also: execute_command, background_command
    """
    p = subprocess.Popen(command, shell=True,
                         stdout=subprocess.PIPE)

    stdout = p.communicate()[0]
    if p.returncode is not 0:
        output = "Command exited with status {0}: {1}".format(
            p.returncode, command)
        if require_zero_status:
            raise Exception(output)
        else:
            logger.warning(output)
    return stdout if type(stdout) is str else stdout.decode()

def wait_for_background_commands():
    """ This waits for all threads to exit.  You will often want to
        run this at the end of programs that have launched background
        threads, so that the program will wait for its child processes
        to terminate before it dies."""
    for t in threading.enumerate():
        if not t == threading.current_thread():
            t.join()

def background_command(command, require_zero_status = False):
    """Executes a command in a separate thread, like running with '&' in the shell.
       If you want the program to die if the command eventually returns with
       nonzero status, then set require_zero_status to True.  'command' will be
       executed in 'shell' mode, so it's OK for it to contain pipes and other
       shell constructs.

       This function returns the Thread object created, just in case you want
       to wait for that specific command to finish.  For example, you could do:
             thread = background_command('foo | bar')
             # do something else while waiting for it to finish
             thread.join()

       See also:
         - wait_for_background_commands(), which can be used
           at the end of the program to wait for all these commands to terminate.
         - execute_command() and get_command_stdout(), which allow you to
           execute commands in the foreground.

    """

    p = subprocess.Popen(command, shell=True)
    thread = threading.Thread(target=background_command_waiter,
                              args=(command, p, require_zero_status))
    thread.daemon=True  # make sure it exits if main thread is terminated
                        # abnormally.
    thread.start()
    return thread


def background_command_waiter(command, popen_object, require_zero_status):
    """ This is the function that is called from background_command, in
        a separate thread."""

    popen_object.communicate()
    if popen_object.returncode is not 0:
        str = "Command exited with status {0}: {1}".format(
            popen_object.returncode, command)
        if require_zero_status:
            logger.error(str)
            # thread.interrupt_main() sends a KeyboardInterrupt to the main
            # thread, which will generally terminate the program.
            thread_module.interrupt_main()
        else:
            logger.warning(str)


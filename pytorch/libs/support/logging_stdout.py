# -*- coding:utf-8 -*-

# Reference: https://stackoverflow.com/questions/1383254/logging-streamhandler-and-standard-streams/55494220#55494220

import sys, logging, threading

def _logging_handle(self, record):
    self.STREAM_LOCKER = getattr(self, "STREAM_LOCKER", threading.RLock())
    if self.stream in (sys.stdout, sys.stderr) and record.levelname in self.FIX_LEVELS:
        try:
            self.STREAM_LOCKER.acquire()
            self.stream = sys.stdout
            self.old_handle(record)
            self.stream = sys.stderr
        finally:
            self.STREAM_LOCKER.release()
    else:
        self.old_handle(record)


def patch_logging_stream(*levels):
    """
    writing some logging level message to sys.stdout

    example:
    patch_logging_stream(logging.INFO, logging.DEBUG)
    logging.getLogger('root').setLevel(logging.DEBUG)

    logging.getLogger('root').debug('test stdout')
    logging.getLogger('root').error('test stderr')
    """
    stream_handler = logging.StreamHandler
    levels = levels or [logging.DEBUG, logging.INFO]
    stream_handler.FIX_LEVELS = [logging.getLevelName(i) for i in levels]
    if hasattr(stream_handler, "old_handle"):
        stream_handler.handle = stream_handler.old_handle
    stream_handler.old_handle = stream_handler.handle
    stream_handler.handle = _logging_handle
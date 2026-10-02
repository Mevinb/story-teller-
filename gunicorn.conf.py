# Gunicorn configuration for Story Teller Creative Suite
import os
import signal
import sys

bind = "127.0.0.1:5000"
workers = 1
threads = 8
worker_class = "gthread"
timeout = 180
graceful_timeout = 5
keepalive = 5

# Direct logs to terminal stdout/stderr so they are immediately visible
accesslog = "-"
errorlog = "-"
capture_output = False
loglevel = "info"

def on_starting(server):
    """
    Ensure the Gunicorn master process terminates automatically if its parent
    (the terminal or launch script) exits or dies unexpectedly.
    """
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass

def post_fork(server, worker):
    """
    Ensure each worker process also dies if the master process dies.
    """
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass

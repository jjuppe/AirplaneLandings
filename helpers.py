import sys
import time
from functools import wraps


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r

    return wrapper


def progress_bar(current, end, title):
    percent = float(current) / end

    sys.stdout.write("\r{0} {1}%".format(title, int(round(percent * 100))))
    sys.stdout.flush()

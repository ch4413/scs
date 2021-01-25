from functools import wraps
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def logger(fn):
    count = 0

    @wraps(fn)
    def inner(*args, **kwargs):
        """
        """
        nonlocal count
        count += 1
        print(f'''Running: Function "{fn.__name__}" \
                (id={id(fn)}) was called {count} times''')
        return fn(*args, **kwargs)

    return inner

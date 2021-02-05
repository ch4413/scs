from functools import wraps
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def logger(fn):
    """
    Summary
    -------
    Wraapped to log excution of functions
    ----------
    fn: python function
        function to be logged

    Returns
    -------
    inner: python function
        wrapped funciton to log functions
    Example
    --------
    @logger.logger
    def <insert function>:
        <define function>
    """
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

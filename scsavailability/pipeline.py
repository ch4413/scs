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
        #print('Running: Function "{0}" (id={1} was called {2} times'.format(fn.__name__, id(fn), count))
        logging.info('Running: Function "{0}" (id={1} was called {2} times'.format(fn.__name__, id(fn), count))
        return fn(*args, **kwargs)
    
    return inner

@logger
def test_function(x):
    y = x**2
    return(y)

value = test_function(4)

print('the value is: {0}'.format(value))

import sys
import random
import numpy as np

from pympler import asizeof

class Utils:

    def flatten(lst):
        # Returns a flatten version of the given list
        # All sublists are merged onto a bigger list. Non-list elements are removed
        return [item for sublst in lst for item in sublst if isinstance(sublst, list)]

    def random_decision(probability):
        # Returns a random True or False depending of the given probability
        return random.random() < probability

    def split(a, n):
        # Split the array a into n evenly sized chunks
        k, m = divmod(len(a), n)
        for i in range(n):
            yield a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]

    def chunks(l, n):
        # Split the array into multiple arrays of size n
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def getsize(obj):
        # Returns the size of the object in bytes (dict for each instance var)
        # Note: not working well with np.memmap
        size  = {}
        size["total"] = 0
        for var, inner_obj in obj.__dict__.items():
            if isinstance(inner_obj, np.core.memmap):
                size[var] = sys.getsizeof(inner_obj)
            else:
                size[var] = asizeof.asizeof(inner_obj)
            size["total"] += size[var]
        return size

    def running_notebook():
        # Returns True if the current code is running in a Jupyter Notebook,
        # False otherwise
        if 'IPython' in sys.modules:
            from IPython import get_ipython
            return 'IPKernelApp' in get_ipython().config
        else:
            return False

    def parse_csv(file_path, delimiter, comment="#"):
        # Returns a list of lines, each line being a list of cells
        output = []
        with open(file_path, "r") as file:
            for line in file.read().splitlines():
                if line[0] == comment:
                    continue
                else:
                    output.append(line.split(delimiter))
        return output

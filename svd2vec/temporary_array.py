
import os
import numpy as np
import tempfile

class TemporaryArray:

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.file_name = tempfile.NamedTemporaryFile().name

    def load(self, erase=False):
        if erase:
            return np.memmap(self.file_name, shape=self.shape, dtype=self.dtype, mode='w+')
        else:
            return np.memmap(self.file_name, shape=self.shape, dtype=self.dtype, mode='r')

    def close(self):
        os.remove(self.file_name)

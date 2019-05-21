
import os
import numpy as np
import tempfile

class TemporaryArray:

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.file_name = tempfile.NamedTemporaryFile().name

        matrix = self.load(erase=True)
        matrix.flush()
        del matrix

    def load(self, erase=False, size=-1, start=0):
        if erase:
            return np.memmap(self.file_name, shape=self.shape, dtype=self.dtype, mode='w+')
        else:
            if size < 0:
                shape = self.shape
            else:
                shape = (size, self.shape[1])
            offset = start * self.shape[1] * self.dtype.itemsize
            #print("On me demande de charger ", shape, " ", offset)
            return np.memmap(self.file_name, shape=shape, dtype=self.dtype, mode='r+', offset=offset)

    def close(self):
        os.remove(self.file_name)

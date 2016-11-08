import luigi
import os
import h5py

class FileTarget(luigi.Target):

    def __init__(self, filename):
        self.filename = filename

    def exists(self):
        return os.path.isfile(self.filename)

class HdfAttributeTarget(luigi.Target):

    def __init__(self, filename, dataset, attribute):
        self.filename = filename
        self.dataset = dataset
        self.attribute = attribute

    def exists(self):
        if not os.path.isfile(self.filename):
            return False
        try:
            with h5py.File(self.filename, 'r') as f:
                return (self.dataset in f and self.attribute in f[self.dataset].attrs)
        except:
            return False

import luigi
import os
import h5py
import json

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

class JsonTarget(luigi.Target):

    def __init__(self, filename, key, value):
        self.filename = filename
        self.key = key
        self.value = value

    def exists(self):
        if not os.path.isfile(self.filename):
            return False
        try:
            with open(self.filename) as f:
                d = json.load(f)
                if not self.key in d:
                    return False
                return self.value == d[self.key]
        except:
            return False

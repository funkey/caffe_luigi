class NoSuchModule(object):

    def __init__(self, name):
        self.name = name

    def __getattr__(item):
        raise ImportError('Module {0} could not be found'.format(self.__name))

try:
    import zwatershed
except ImportError:
    zwatershed = NoSuchModule('zwatershed')

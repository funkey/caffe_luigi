from contextlib import contextmanager
import errno
import fcntl
import numbers
import os
import signal

pools = {}

@contextmanager
def timeout(seconds):

    def timeout_handler(signum, frame):
        pass

    original_handler = signal.signal(signal.SIGALRM, timeout_handler)

    try:
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

class OutOfSharedResources(Exception):
    pass

class Resource:
    '''Represents a resource with an id. On deletion of this object, the 
    resource with this id is freed.'''

    def __init__(self, fd, id):
        self.__fd = fd
        self.id = id

class ResourcePool:

    def resource_dir(self):
        return os.path.join(os.path.expanduser('~'),'.luigi', 'resources', self.name)

    def __init__(self, name, instances):
        self.name = name
        self.instances = instances
        if not os.path.isdir(self.resource_dir()):
            os.makedirs(self.resource_dir())

    def get(self):
        '''Try to lock a resource and return it.'''

        for i in self.instances:
            with timeout(1):
                fd = open(os.path.join(self.resource_dir(),str(i)), 'w')
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    # this resource is available
                    return Resource(fd, i)
                except IOError, e:
                    if e.errno != errno.EINTR:
                        raise e
                    # this resource is locked, continue

        # no available resource was found
        raise OutOfSharedResources('no more available resource with name ' + self.name)

def register_shared_resource(name, instances):
    '''Register a named resource, which has the given instances are available.

    The instances have to be convertable to a string, e.g., a list of intergers is fine.'''
    global pools
    pools[name] = ResourcePool(name, instances)

def lock(name):
    '''Attempt to lock a shared resource. On success, a Resource instance is 
    returned, which guarantees exclusivity of the resource. If no more shared resource is available, `OutOfSharedResources` is thrown.'''

    return pools[name].get()

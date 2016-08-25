import luigi
from shared_resource import *
from tasks import *
from targets import *

if __name__ == '__main__':
    register_shared_resource('gpu', 2)
    luigi.run(main_task_cls=SegmentAllTask)

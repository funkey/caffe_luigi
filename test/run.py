import sys
sys.path.append('..')
import luigi
from shared_resource import *
from tasks import *
from targets import *

from create_segmentations import create_segmentations
class Test(luigi.Task):
    def run(self):
        with redirect_output(self):
            create_segmentations(1, 2, 3, 4, 5)

if __name__ == '__main__':
    register_shared_resource('gpu', 2)
    luigi.run(main_task_cls=Test)

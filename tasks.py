import luigi
import time
import random
from targets import *
from shared_resource import *

class ProcessTask(luigi.Task):

    setup = luigi.IntParameter()
    resources = { 'gpu':1 }

    def output_filename(self):
        return 'data/affinities.%d.dat'%self.setup

    def output(self):
        return FileTarget(self.output_filename())

    def run(self):
        gpu = lock('gpu')
        print('=============== processing with gpu ' + str(gpu.id))
        time.sleep(10)
        with open(self.output_filename(), 'w') as f:
            f.write('initial file')

class SegmentTask(luigi.Task):

    threshold = luigi.IntParameter()
    setup = luigi.IntParameter()

    def output_filename(self):
        return 'data/affinities.%d.%d.dat'%(self.setup,self.threshold)

    def requires(self):
        return ProcessTask(self.setup)

    def output(self):
        return FileTarget(self.output_filename())

    def run(self):
        print("Starting SegmentTask for threshold " + str(self.threshold) + ", setup " + str(self.setup))
        random.seed(time.time())
        if random.random() > 0.5:
            print("Whoops!")
            raise RuntimeError("Something went wrong")
            #return # without producing results
        time.sleep(2)
        with open(self.output_filename(), 'w') as f:
            f.write('done')
        print("SegmentTask for threshold " + str(self.threshold) + " finished")

class SegmentAllTask(luigi.task.WrapperTask):

    def requires(self):
        return [ SegmentTask(t,s) for t in range(100, 1000, 100) for s in range(8) ]

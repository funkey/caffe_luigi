import glob
import luigi
import os
import socket
import sys
from redirect_output import redirect_output
from shared_resource import *
from targets import *

base_dir = '.'
def set_base_dir(d):
    global base_dir
    base_dir = d

class TrainTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()

    resources = { 'gpu_{}'.format(socket.gethostname()) :1 }

    def output_filename(self):
        return os.path.join(base_dir, '02_train', str(self.setup), 'net_iter_%d.solverstate'%self.iteration)

    def requires(self):
        if self.iteration == 2000:
            return []
        return TrainTask(self.experiment, self.setup, self.iteration - 2000)

    def output(self):
        return FileTarget(self.output_filename())

    def run(self):
        with redirect_output(self):
            gpu = lock('gpu_{}'.format(socket.gethostname()))
            os.chdir(os.path.join(base_dir, '02_train', self.setup))
            sys.path.append(os.getcwd())
            from train_until import train_until
            train_until(self.iteration, gpu.id)

class ProcessTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()
    augmentation = luigi.Parameter()

    resources = { 'gpu_{}'.format(socket.gethostname()) :1 }

    def output_filename(self):
        self.augmentation_suffix = ''
        if self.augmentation is not None:
            self.augmentation_suffix = '.augmented.%d'%self.augmentation
        if '+' in self.sample:
            process_dir = '04_process_testing'
        else:
            process_dir = '03_process_training'
        return os.path.join(base_dir, process_dir, 'processed', self.setup, str(self.iteration), '%s%s.hdf'%(self.sample,self.augmentation_suffix))

    def requires(self):
        return TrainTask(self.experiment, self.setup, self.iteration)

    def output(self):
        return FileTarget(self.output_filename())

    def run(self):
        with redirect_output(self):
            gpu = lock('gpu_{}'.format(socket.gethostname()))
            from predict_affinities import predict_affinities
            predict_affinities(self.setup, self.iteration, self.sample, self.augmentation, gpu=gpu.id)

class SegmentTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()
    augmentation = luigi.Parameter()
    threshold = luigi.IntParameter()

    resources = { 'segment_task_count_{}'.format(socket.gethostname()) :1 }

    def get_setup(self):
        if isinstance(self.setup, int):
            return 'setup%02d'%self.setup
        return self.setup

    def get_iteration(self):
        if self.iteration == -1:
            # take the most recent iteration
            modelfiles = glob.glob(os.path.join(base_dir, '02_train', str(self.get_setup()), 'net_iter_*.solverstate'))
            iterations = [ int(modelfile.split('_')[-1].split('.')[0]) for modelfile in modelfiles ]
            self.iteration = max(iterations)
        return self.iteration

    def output_filename(self, threshold):
        self.augmentation_suffix = ''
        if self.augmentation is not None:
            self.augmentation_suffix = '.augmented.%d'%self.augmentation
        if '+' in self.sample:
            process_dir = '04_process_testing'
        else:
            process_dir = '03_process_training'
        return os.path.join(base_dir, process_dir, 'processed', self.get_setup(), str(self.iteration), '%s%s.%d.hdf'%(self.sample,self.augmentation_suffix,threshold))

    def requires(self):
        return ProcessTask(self.experiment, self.get_setup(), self.get_iteration(), self.sample, self.augmentation)

    def output(self):
        return FileTarget(self.output_filename(self.threshold))

    def run(self):
        with redirect_output(self):
            from create_segmentations import create_segmentations
            create_segmentations(self.get_setup(), self.get_iteration(), self.sample, self.augmentation, [self.threshold])

class EvaluateTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()
    augmentation = luigi.Parameter()
    threshold = luigi.IntParameter()

    def get_setup(self):
        if isinstance(self.setup, int):
            return 'setup%02d'%self.setup
        return self.setup

    def get_iteration(self):
        if self.iteration == -1:
            # take the most recent iteration
            modelfiles = glob.glob(os.path.join(base_dir, '02_train', str(self.get_setup()), 'net_iter_*.solverstate'))
            iterations = [ int(modelfile.split('_')[-1].split('.')[0]) for modelfile in modelfiles ]
            self.iteration = max(iterations)
        return self.iteration

    def output_filename(self):
        self.augmentation_suffix = ''
        if self.augmentation is not None:
            self.augmentation_suffix = '.augmented.%d'%self.augmentation
        if '+' in self.sample:
            process_dir = '04_process_testing'
        else:
            process_dir = '03_process_training'
        return os.path.join(base_dir, process_dir, 'processed', self.get_setup(), str(self.iteration), '%s%s.%d.hdf'%(self.sample,self.augmentation_suffix,self.threshold))

    def requires(self):
        return SegmentTask(self.experiment, self.get_setup(), self.get_iteration(), self.sample, self.augmentation, self.threshold)

    def output(self):
        return HdfAttributeTarget(self.output_filename(), 'main', 'evaluated')

    def run(self):
        with redirect_output(self):
            from evaluate import evaluate
            evaluate(self.output_filename())

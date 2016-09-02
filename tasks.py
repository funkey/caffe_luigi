import glob
import luigi
import os
import socket
import sys
from redirect_output import redirect_output
from shared_resource import *
from targets import *

class TrainTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()

    resources = { 'gpu':1 }

    def output_filename(self):
        return '%s/net_iter_%d.solverstate'%(self.setup,self.iteration)

    def requires(self):
        if self.iteration == 2000:
            return []
        return TrainTask(self.experiment, self.setup, self.iteration - 2000)

    def output(self):
        return FileTarget(self.output_filename())

    def run(self):
        with redirect_output(self):
            gpu = lock('gpu')
            os.chdir(self.setup)
            sys.path.append(os.getcwd())
            from train_until import train_until
            train_until(self.iteration, gpu.id)

class ProcessTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()
    augmentation = luigi.Parameter()

    resources = { 'gpu':1 }

    def output_filename(self):
        self.augmentation_suffix = ''
        if self.augmentation is not None:
            self.augmentation_suffix = '.augmented.%d'%self.augmentation
        return 'processed/%s/%d/%s_less-padded_20160501%s.hdf'%(self.setup,self.iteration,self.sample,self.augmentation_suffix)

    def requires(self):
        return TrainTask(self.setup, self.iteration)

    def output(self):
        return FileTarget(self.output_filename())

    def run(self):
        with redirect_output(self):
            gpu = lock('gpu')
            from predict_affinities import predict_affinities
            predict_affinities(self.setup, self.iteration, self.sample, self.augmentation, gpu=gpu.id)

class SegmentTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()
    augmentation = luigi.Parameter()
    thresholds = luigi.Parameter()

    resources = { 'segment_task_count_{}'.format(socket.gethostname()) :1 }

    def get_setup(self):
        if isinstance(self.setup, int):
            return 'setup%02d'%self.setup
        return self.setup

    def get_iteration(self):
        if self.iteration == -1:
            # take the most recent iteration
            modelfiles = glob.glob('../02_train/%s/net_iter_*.solverstate'%self.get_setup())
            iterations = [ int(modelfile.split('_')[-1].split('.')[0]) for modelfile in modelfiles ]
            self.iteration = max(iterations)
        return self.iteration

    def output_filename(self, threshold):
        self.augmentation_suffix = ''
        if self.augmentation is not None:
            self.augmentation_suffix = '.augmented.%d'%self.augmentation
        return 'processed/%s/%d/%s_less-padded_20160501%s.%d.hdf'%(self.get_setup(),self.get_iteration(),self.sample,self.augmentation_suffix,threshold)

    def requires(self):
        return ProcessTask(self.experiment, self.get_setup(), self.get_iteration(), self.sample, self.augmentation)

    def output(self):
        return [ FileTarget(self.output_filename(t)) for t in self.thresholds ]

    def run(self):
        with redirect_output(self):
            from create_segmentations import create_segmentations
            create_segmentations(self.get_setup(), self.get_iteration(), self.sample, self.augmentation, self.thresholds)

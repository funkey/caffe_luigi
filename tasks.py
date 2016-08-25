import luigi
import time
import random
from targets import *
from shared_resource import *
from predict_affinities import predict_affinities
from create_segmentations import create_segmentations
from redirect_output import redirect_output

class TrainTask(luigi.task.ExternalTask):

    setup = luigi.IntParameter()
    iteration = luigi.IntParameter()

    def output_filename(self):
        return '../02_train/setup%02d/net_iter_%d.solverstate'%(self.setup,self.iteration)

    def output(self):
        return FileTarget(self.output_filename())

class ProcessTask(luigi.Task):

    setup = luigi.IntParameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()
    augmentation = luigi.Parameter()

    resources = { 'gpu':1 }

    def output_filename(self):
        self.augmentation_suffix = ''
        if self.augmentation is not None:
            self.augmentation_suffix = '.augmented.%d'%self.augmentation
        return 'processed/setup%02d/%d/%s_less-padded_20160501%s.hdf'%(self.setup,self.iteration,self.sample,self.augmentation_suffix)

    def output(self):
        return FileTarget(self.output_filename())

    def run(self):
        gpu = lock('gpu')
        with redirect_output(self):
            predict_affinities(self.setup, self.iteration, self.sample, self.augmentation, gpu=gpu.id)

class SegmentTask(luigi.Task):

    setup = luigi.IntParameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()
    augmentation = luigi.Parameter()
    thresholds = luigi.Parameter()

    resources = { 'segment_task_count':1 }

    def output_filename(self, threshold):
        self.augmentation_suffix = ''
        if self.augmentation is not None:
            self.augmentation_suffix = '.augmented.%d'%self.augmentation
        return 'processed/setup%02d/%d/%s_less-padded_20160501%s.%d.hdf'%(self.setup,self.iteration,self.sample,self.augmentation_suffix,threshold)

    def requires(self):
        return ProcessTask(self.setup, self.iteration, self.sample, self.augmentation)

    def output(self):
        return [ FileTarget(self.output_filename(t)) for t in self.thresholds ]

    def run(self):
        with redirect_output(self):
            create_segmentations(self.setup, self.iteration, self.sample, self.augmentation, self.thresholds)

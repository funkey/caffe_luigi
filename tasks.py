import glob
import luigi
import os
import socket
import sys
import waterz
from redirect_output import *
from shared_resource import *
from targets import *

base_dir = '.'
def set_base_dir(d):
    global base_dir
    base_dir = d

class RunTasks(luigi.WrapperTask):
    '''Top-level task to run several tasks.'''

    tasks = luigi.Parameter()

    def requires(self):
        return self.tasks

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
        log_base = os.path.join(base_dir, '02_train', str(self.setup), 'train_%d'%self.iteration)
        with RedirectOutput(log_base + '.out', log_base + '.err'):
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

    def process_dir(self):
        if '+' in self.sample:
            return '04_process_testing'
        else:
            return '03_process_training'

    def output_filename(self):
        self.augmentation_suffix = ''
        if self.augmentation is not None:
            self.augmentation_suffix = '.augmented.%d'%self.augmentation
        return os.path.join(base_dir, self.process_dir(), 'processed', self.setup, str(self.iteration), '%s%s.hdf'%(self.sample,self.augmentation_suffix))

    def requires(self):
        return TrainTask(self.experiment, self.setup, self.iteration)

    def output(self):
        return FileTarget(self.output_filename())

    def run(self):
        log_base = os.path.join(base_dir, self.process_dir(), 'processed', self.setup, str(self.iteration), '%s%s'%(self.sample,self.augmentation_suffix))
        with RedirectOutput(log_base + '.out', log_base + '.err'):
            gpu = lock('gpu_{}'.format(socket.gethostname()))
            from predict_affinities import predict_affinities
            predict_affinities(self.setup, self.iteration, self.sample, self.augmentation, gpu=gpu.id)

class ChunkProcessTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()
    data_dir = luigi.Parameter()
    chunk_offset = luigi.Parameter()
    chunk_size = luigi.Parameter()

    resources = { 'gpu_{}'.format(socket.gethostname()) :1 }

    def output_basename(self):
        return os.path.join('.', 'processed', self.setup, str(self.iteration), '%s_%s_%s'%(self.sample,str(self.chunk_offset),str(self.chunk_size)))

    def requires(self):
        return TrainTask(self.experiment, self.setup, self.iteration)

    def output(self):
        return FileTarget(self.output_basename() + '.hdf')

    def run(self):
        log_base = self.output_basename()
        with RedirectOutput(log_base + '.out', log_base + '.err'):
            gpu = lock('gpu_{}'.format(socket.gethostname()))
            from predict_affinities import predict_affinities
            chunk = { 'offset': self.chunk_offset, 'size': self.chunk_size }
            predict_affinities(self.setup, self.iteration, self.sample, augmentation=None, gpu=gpu.id, orig_data_dir=self.data_dir, chunk=chunk)

class Evaluate(luigi.Task):

    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()
    augmentation = luigi.Parameter()

    experiment = luigi.Parameter()
    thresholds = luigi.Parameter()
    custom_fragments = luigi.BoolParameter()
    histogram_quantiles = luigi.BoolParameter()
    discrete_queue = luigi.BoolParameter()
    merge_function = luigi.Parameter()
    dilate_mask = luigi.IntParameter(default=0)
    mask_fragments = luigi.IntParameter(default=False)

    aff_high = luigi.Parameter()
    aff_low = luigi.Parameter()

    keep_segmentation = luigi.BoolParameter()

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

    def process_dir(self):
        if '+' in self.sample:
            return '04_process_testing'
        else:
            return '03_process_training'

    def tag(self):
        tag = self.sample + '_' + self.merge_function
        if self.merge_function != 'zwatershed' and waterz.__version__ != '0.6':
            tag += '_' + waterz.__version__
        if self.custom_fragments:
            tag += '_cf'
        elif self.merge_function == 'zwatershed': # only for 'zwatershed', for all other ones we use the default values
            tag += '_ah%f_al%f'%(self.aff_high,self.aff_low)
        if self.histogram_quantiles:
            tag += '_hq'
        if self.discrete_queue:
            tag += '_dq'
        if self.dilate_mask != 0:
            tag += '_dm%d'%self.dilate_mask
        if self.mask_fragments:
            tag += '_mf'
        if self.augmentation is not None:
            tag += '_au%d'%self.augmentation
        return tag

    def output_basename(self, threshold=None):

        threshold_string = ''
        if threshold is not None:
            threshold_string = ('%f'%threshold).rstrip('0').rstrip('.')
        basename = self.tag() + threshold_string

        return os.path.join(
                base_dir,
                self.process_dir(),
                'processed',
                self.get_setup(),
                str(self.iteration),
                basename)

    def requires(self):
        return ProcessTask(self.experiment, self.get_setup(), self.get_iteration(), self.sample, self.augmentation)

    def output(self):
        targets = [ JsonTarget(self.output_basename(t) + '.json', 'setup', self.get_setup()) for t in self.thresholds ]
        if self.keep_segmentation:
            targets += [ FileTarget(self.output_basename(t) + '.hdf') for t in self.thresholds ]
        return targets

    def run(self):
        with RedirectOutput(self.output_basename() + '.out', self.output_basename() + '.err'):
            from evaluate import evaluate
            evaluate(
                    setup=self.setup,
                    iteration=self.iteration,
                    sample=self.sample,
                    augmentation=self.augmentation,
                    thresholds=self.thresholds,
                    output_basenames=[self.output_basename(t) for t in self.thresholds],
                    custom_fragments=self.custom_fragments,
                    histogram_quantiles=self.histogram_quantiles,
                    discrete_queue=self.discrete_queue,
                    merge_function=self.merge_function,
                    dilate_mask=self.dilate_mask,
                    mask_fragments=self.mask_fragments,
                    keep_segmentation=self.keep_segmentation,
                    aff_high=self.aff_high,
                    aff_low=self.aff_low)

class EvaluateIteration(luigi.task.WrapperTask):

    setup = luigi.Parameter()
    iteration = luigi.Parameter()
    samples = luigi.Parameter()
    augmentations = luigi.Parameter()
    parameters = luigi.Parameter()

    @property
    def priority(self):
        # process largely spaced iterations first
        if self.iteration % 100000 == 0:
            return 10
        elif self.iteration % 50000 == 0:
            return 5
        elif self.iteration % 10000 == 0:
            return 3
        return 0

    def requires(self):

        return [
            Evaluate(
                setup=self.setup,
                iteration=self.iteration,
                sample=sample,
                augmentation=augmentation,
                **self.parameters)
            for sample in self.samples
            for augmentation in self.augmentations
        ]

class EvaluateSetup(luigi.task.WrapperTask):

    setup = luigi.Parameter()
    iterations = luigi.Parameter()
    samples = luigi.Parameter()
    augmentations = luigi.Parameter()
    parameters = luigi.Parameter()

    def requires(self):

        return [
            EvaluateIteration(
                setup=self.setup,
                iteration=iteration,
                samples=self.samples,
                augmentations=self.augmentations,
                parameters=self.parameters)
            for iteration in self.iterations
        ]

class EvaluateConfiguration(luigi.task.WrapperTask):

    setups = luigi.Parameter()
    iterations = luigi.Parameter()
    samples = luigi.Parameter()
    augmentations = luigi.Parameter()
    parameters = luigi.Parameter()

    def requires(self):

        return [
            EvaluateSetup(
                setup=setup,
                iterations=self.iterations,
                samples=self.samples,
                augmentations=self.augmentations,
                parameters=self.parameters)
            for setup in self.setups
        ]

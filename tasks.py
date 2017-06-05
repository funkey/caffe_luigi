import glob
import luigi
import os
import socket
import sys
import waterz
from redirect_output import *
from shared_resource import *
from targets import *
from subprocess import call

base_dir = '.'
def set_base_dir(d):
    global base_dir
    base_dir = d

def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass

class RunTasks(luigi.WrapperTask):
    '''Top-level task to run several tasks.'''

    tasks = luigi.Parameter()

    def requires(self):
        return self.tasks

class TrainTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()

    # resources = { 'gpu_{}'.format(socket.gethostname()) :1 }

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
        log_out = log_base + '.out'
        log_err = log_base + '.err'
        os.chdir(os.path.join(base_dir, '02_train', self.setup))
        call([
            'run_mesos.sh',
            '-c', '10',
            '-g', '1',
            '-d', 'funkey/gunpowder:v0.2-prerelease',
            '-e', 'python -u train_until.py ' + str(self.iteration) + ' 0 1>%s 2>%s'%(log_out,log_err)
        ])

class ProcessTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()

    # resources = { 'gpu_{}'.format(socket.gethostname()) :1 }

    def output_filename(self):
        return os.path.join(base_dir, '03_process', 'processed', self.setup, str(self.iteration), '%s.hdf'%self.sample)

    def requires(self):
        return TrainTask(self.experiment, self.setup, self.iteration)

    def output(self):
        return FileTarget(self.output_filename())

    def run(self):
        mkdirs(os.path.join(base_dir, '03_process', 'processed', self.setup, str(self.iteration)))
        log_base = os.path.join(base_dir, '03_process', 'processed', self.setup, str(self.iteration), '%s'%self.sample)
        log_out = log_base + '.out'
        log_err = log_base + '.err'
        os.chdir(os.path.join(base_dir, '03_process'))
        call([
            'run_mesos.sh',
            '-c', '5',
            '-g', '1',
            '-d', 'funkey/gunpowder:v0.2-prerelease',
            '-e', 'python -u predict_affinities.py ' + self.setup + ' ' + str(self.iteration) + ' ' + self.sample + ' 0 1>%s 2>%s'%(log_out,log_err)
        ])

class Evaluate(luigi.Task):

    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()

    experiment = luigi.Parameter()
    thresholds = luigi.Parameter()
    custom_fragments = luigi.BoolParameter()
    histogram_quantiles = luigi.BoolParameter()
    discrete_queue = luigi.BoolParameter()
    merge_function = luigi.Parameter()
    init_with_max = luigi.Parameter()
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
        if self.init_with_max:
            tag += '_im'
        return tag

    def output_basename(self, threshold=None):

        threshold_string = ''
        if threshold is not None:
            threshold_string = ('%f'%threshold).rstrip('0').rstrip('.')
        basename = self.tag() + threshold_string

        return os.path.join(
                base_dir,
                '03_process',
                'processed',
                self.get_setup(),
                str(self.iteration),
                basename)

    def requires(self):
        return ProcessTask(self.experiment, self.get_setup(), self.get_iteration(), self.sample)

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
                    thresholds=self.thresholds,
                    output_basenames=[self.output_basename(t) for t in self.thresholds],
                    custom_fragments=self.custom_fragments,
                    histogram_quantiles=self.histogram_quantiles,
                    discrete_queue=self.discrete_queue,
                    merge_function=self.merge_function,
                    init_with_max=self.init_with_max,
                    dilate_mask=self.dilate_mask,
                    mask_fragments=self.mask_fragments,
                    keep_segmentation=self.keep_segmentation,
                    aff_high=self.aff_high,
                    aff_low=self.aff_low)

class EvaluateIteration(luigi.task.WrapperTask):

    setup = luigi.Parameter()
    iteration = luigi.Parameter()
    samples = luigi.Parameter()
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
                **self.parameters)
            for sample in self.samples
        ]

class EvaluateSetup(luigi.task.WrapperTask):

    setup = luigi.Parameter()
    iterations = luigi.Parameter()
    samples = luigi.Parameter()
    parameters = luigi.Parameter()

    def requires(self):

        return [
            EvaluateIteration(
                setup=self.setup,
                iteration=iteration,
                samples=self.samples,
                parameters=self.parameters)
            for iteration in self.iterations
        ]

class EvaluateConfiguration(luigi.task.WrapperTask):

    setups = luigi.Parameter()
    iterations = luigi.Parameter()
    samples = luigi.Parameter()
    parameters = luigi.Parameter()

    def requires(self):

        return [
            EvaluateSetup(
                setup=setup,
                iterations=self.iterations,
                samples=self.samples,
                parameters=self.parameters)
            for setup in self.setups
        ]

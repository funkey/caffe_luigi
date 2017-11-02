import glob
import luigi
import os
import socket
import sys
import waterz
import itertools
import json
from redirect_output import *
from shared_resource import *
from targets import *
from subprocess import check_call

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

    def output_filename(self):
        return os.path.join(
            base_dir,
            '02_train',
            str(self.setup),
            'unet_checkpoint_%d.meta'%self.iteration)

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
        with open(log_out, 'w') as o:
            with open(log_err, 'w') as e:
                check_call([
                    'run_slurm',
                    '-c', '10',
                    '-g', '1',
                    '-d', 'funkey/gunpowder:v0.3-pre5',
                    'python -u train_until.py ' + str(self.iteration) + ' 0'
                ], stdout=o, stderr=e)

class ProcessTask(luigi.Task):

    experiment = luigi.Parameter()
    setup = luigi.Parameter()
    iteration = luigi.IntParameter()
    sample = luigi.Parameter()

    def output_filename(self):
        return os.path.join(base_dir, '03_process', 'processed', self.setup, str(self.iteration), '%s.hdf'%self.sample)

    def requires(self):
        return TrainTask(self.experiment, self.setup, self.iteration)

    def output(self):
        return HdfDatasetTarget(self.output_filename(), 'volumes/predicted_affs')

    def run(self):
        mkdirs(os.path.join(base_dir, '03_process', 'processed', self.setup, str(self.iteration)))
        log_base = os.path.join(base_dir, '03_process', 'processed', self.setup, str(self.iteration), '%s'%self.sample)
        log_out = log_base + '.out'
        log_err = log_base + '.err'
        os.chdir(os.path.join(base_dir, '03_process'))
        with open(log_out, 'w') as o:
            with open(log_err, 'w') as e:
                check_call([
                    'run_slurm',
                    '-c', '2',
                    '-g', '1',
                    '-m', '350000',
                    '-d', 'funkey/gunpowder:v0.3-pre5',
                    'python -u predict_affinities.py ' + self.setup + ' ' + str(self.iteration) + ' ' + self.sample + ' 0'
                ], stdout=o, stderr=e)

class Evaluate(luigi.Task):

    parameters = luigi.DictParameter()

    def get_setup(self):
        if isinstance(self.parameters['setup'], int):
            return 'setup%02d'%self.parameters['setup']
        return self.parameters['setup']

    def get_iteration(self):
        return self.parameters['iteration']

    def tag(self):
        tag = self.parameters['sample'] + '_' + self.parameters['merge_function']
        if self.parameters['merge_function'] != 'zwatershed' and waterz.__version__ != '0.6':
            tag += '_' + waterz.__version__
        if self.parameters['custom_fragments']:
            tag += '_cf'
        elif self.parameters['merge_function'] == 'zwatershed': # only for 'zwatershed', for all other ones we use the default values
            tag += '_ah%f_al%f'%(self.parameters['aff_high'],self.parameters['aff_low'])
        if self.parameters['histogram_quantiles']:
            tag += '_hq'
        if self.parameters['discrete_queue']:
            tag += '_dq'
        if self.parameters['dilate_mask'] != 0:
            tag += '_dm%d'%self.parameters['dilate_mask']
        if self.parameters['mask_fragments']:
            tag += '_mf'
        if self.parameters['init_with_max']:
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
                str(self.get_iteration()),
                basename)

    def requires(self):
        return ProcessTask(
            self.parameters['experiment'],
            self.get_setup(),
            self.get_iteration(),
            self.parameters['sample'])

    def output(self):

        targets = [
            JsonTarget(self.output_basename(t) + '.json', 'setup', self.get_setup())
            for t in self.parameters['thresholds'] ]
        if self.parameters['keep_segmentation']:
            targets += [
                FileTarget(self.output_basename(t) + '.hdf')
                for t in self.parameters['thresholds'] ]
        return targets

    def run(self):

        # skip invalid configurations
        if self.parameters['merge_function'] == 'mean_aff':
            if self.parameters['init_with_max']:
                return
            if self.parameters['histogram_quantiles']:
                return

        log_out = self.output_basename() + '.out'
        log_err = self.output_basename() + '.err'
        args = dict(self.parameters)
        args['output_basenames'] = [
            self.output_basename(t)
            for t in self.parameters['thresholds']]
        with open(self.output_basename() + '.config', 'w') as f:
            json.dump(args, f)
        with open(log_out, 'w') as o:
            with open(log_err, 'w') as e:
                check_call([
                    'run_slurm',
                    '-c', '2',
                    '-m', '100000',
                    'python -u ../src/caffe_luigi/evaluate.py ' + self.output_basename() + '.config'
                ], stdout=o, stderr=e)

class EvaluateCombinations(luigi.task.WrapperTask):

    # a dictionary containing lists of parameters to evaluate
    parameters = luigi.DictParameter()
    range_keys = luigi.ListParameter()

    def requires(self):

        for k in self.range_keys:
            assert len(k) > 0 and k[-1] == 's', ("Explode keys have to end in "
                                                 "a plural 's'")

        # get all the values to explode
        range_values = {
            k[:-1]: v
            for k, v in self.parameters.iteritems()
            if k in self.range_keys }

        other_values = {
            k: v
            for k, v in self.parameters.iteritems()
            if k not in self.range_keys }

        range_keys = range_values.keys()
        tasks = []
        for concrete_values in itertools.product(*list(range_values.values())):

            parameters = { k: v for k, v in zip(range_keys, concrete_values) }
            parameters.update(other_values)

            tasks.append(Evaluate(parameters))

        return tasks

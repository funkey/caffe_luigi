from itertools import izip
from scipy import ndimage as ndi
import cremi
import h5py
import json
import mahotas as mh
import numpy as np
import os
import time
import waterz
import waterz_aff_squared

# in nm, equivalent to CREMI metric
neuron_ids_border_threshold = 25

def threshold_cc(membrane, threshold):

    thresholded = membrane<=threshold
    return ndi.label(thresholded)[0]

def create_boundary_map_watersheds(affs, seg_thresholds):

    # pixel-wise predictions = average of x and y affinity
    print "Computing membrane map"
    #membrane = 1.0 - (affs[0] + affs[1] + affs[2])/3
    membrane = 1.0 - (affs[1] + affs[2])/2

    segs = []
    for t in seg_thresholds:

        print "Processing threshold " + str(t)

        print "Finding initial seeds"
        seeds = threshold_cc(membrane, t)

        print "Watershedding"
        #segs.append(seeds)
        segs.append(np.array(mh.cwatershed(membrane, seeds), dtype=np.uint64))

    return segs

def create_watersheds(affs, gt, seg_thresholds, treat_as_boundary_map = False, tag = None):

    print "Computing watersheds"
    if treat_as_boundary_map:
        return create_boundary_map_watersheds(affs, seg_thresholds)
    else:
        if tag == 'waterz_aff_squared':
            # aff_squared scoring function produces much higher values
            scaled_thresholds = [ 100*t for t in seg_thresholds ]
            return waterz_aff_squared.agglomerate(affs, scaled_thresholds, gt)
        else:
            return waterz.agglomerate(affs, seg_thresholds, gt)

def crop(a, bb):

    cur_shape = list(a.shape[-3:])
    print("Cropping from " + str(cur_shape) + " to " + str(bb))
    if len(a.shape) == 3:
        a = a[bb]
    elif len(a.shape) == 4:
        a = a[(slice(0,4),)+bb]
    else:
        raise RuntimeError("encountered array of dimension " + str(len(a.shape)))

    return a

def get_gt_bounding_box(gt):

    # no-label ids are <0, i.e. the highest numbers in uint64
    fg_indices = np.where(gt <= np.uint64(-10))
    return tuple(
        slice(np.min(fg_indices[d]),np.max(fg_indices[d])+1)
        for d in range(3)
    )

def is_testing_sample(sample):

    if '+' in sample:
        return True
    return False

def evaluate(setup, iteration, sample, augmentation, seg_thresholds, output_files, treat_as_boundary_map = False, tag = None, keep_segmentation = False):

    if isinstance(setup, int):
        setup = 'setup%02d'%setup

    seg_thresholds = list(seg_thresholds)

    augmentation_name = sample
    if augmentation is not None:
        augmentation_name += '.augmented.' + str(augmentation)

    aff_data_dir = os.path.join(os.getcwd(), 'processed', setup, str(iteration))
    aff_filename = os.path.join(aff_data_dir, augmentation_name + ".hdf")

    print "Evaluating " + sample + " with " + setup + ", iteration " + str(iteration) + " at thresholds " + str(seg_thresholds)

    print "Reading ground truth..."
    if is_testing_sample(sample):
        orig_data_dir = '../00_dataset_preparation/test/'
    else:
        with open(os.path.join('../02_train/', setup, 'train_options.json'), 'r') as f:
            orig_data_dir = json.load(f)['data_dir']
    orig_filename = os.path.join(orig_data_dir, augmentation_name + '.hdf')

    print "Reading ground-truth..."
    if 'resolution' not in h5py.File(orig_filename, 'r')['volumes/labels/neuron_ids'].attrs:
        # this is lost in the alignment/augmentation
        h5py.File(orig_filename, 'r+')['volumes/labels/neuron_ids'].attrs['resolution'] = (40,4,4)
    truth = cremi.io.CremiFile(orig_filename, 'r')
    gt_volume = truth.read_neuron_ids()

    print "Getting ground-truth bounding box..."
    bb = get_gt_bounding_box(gt_volume.data)

    print "Cropping ground-truth to label bounding box"
    gt_volume.data = crop(gt_volume.data, bb)

    print "Growing ground-thruth boundary..."
    gt_volume.data[gt_volume.data>=np.uint64(-10)] = -1
    print("GT min/max: " + str(gt_volume.data.min()) + "/" + str(gt_volume.data.max()))
    evaluate = cremi.evaluation.NeuronIds(gt_volume, border_threshold=neuron_ids_border_threshold)
    gt_with_borders = np.array(evaluate.gt, dtype=np.uint32)
    print("GT with border min/max: " + str(gt_with_borders.min()) + "/" + str(gt_with_borders.max()))

    print "Reading affinities..."
    aff_file = h5py.File(aff_filename, 'r')
    affs = aff_file['main']

    print "Cropping affinities to ground-truth bounding box..."
    affs = crop(affs, bb)

    print "Copying affs to memory..."
    # for waterz
    affs = np.array(affs)
    aff_file.close()

    start = time.time()

    i = 0
    for seg_metric in create_watersheds(affs, gt_with_borders, seg_thresholds, treat_as_boundary_map, tag=tag):

        output_file = output_files[i]

        if keep_segmentation:

            print "Storing segmentation..."

            seg = seg_metric[0]
            h5py.File(output_file, 'w')['main'] = seg

        else:

            print "Storing record..."

            metrics = seg_metric[1]
            threshold = seg_thresholds[i]
            i += 1

            print seg_metric
            print metrics

            record = {
                'setup': setup,
                'iteration': iteration,
                'sample': sample,
                'augmentation': augmentation,
                'threshold': threshold,
                'tag': tag,
                'raw': { 'filename': orig_filename, 'dataset': 'volumes/raw' },
                'gt': { 'filename': orig_filename, 'dataset': 'volumes/labels/gt' },
                'affinities': { 'filename': aff_filename, 'dataset': 'main' },
                'voi_split': metrics['V_Info_split'],
                'voi_merge': metrics['V_Info_merge'],
                'rand_split': metrics['V_Rand_split'],
                'rand_merge': metrics['V_Rand_merge'],
                'gt_border_threshold': neuron_ids_border_threshold
            }
            with open(output_file, 'w') as f:
                json.dump(record, f)


    print "Finished waterz in " + str(time.time() - start) + "s"

if __name__ == "__main__":

    evaluate('setup26', 200000, 'sample_A', 0, [100,200], ['test1.json', 'test2.json'], treat_as_boundary_map = False)

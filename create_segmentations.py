import h5py
import json
import math
import numpy as np
import os
import sys
import zwatershed as zw
import time

def show_stats(v, name):
    print name + ":"
    print "\tshape :" + str(v.shape)
    print "\tdtype :" + str(v.dtype)

def seg_filename(aff_filename, threshold):
    return aff_filename[:-3] + str(threshold) + ".hdf"

def create_watersheds(affs, seg_thresholds, aff_filename):

    print "Computing watersheds"
    segs = zw.zwatershed(affs, seg_thresholds)

    for i in range(len(seg_thresholds)):
        t = seg_thresholds[i]
        print "Storing segmentation"
        seg_file = h5py.File(seg_filename(aff_filename, t), 'w')
        ds = 'main'
        if ds in seg_file:
            del seg_file[ds]
        seg_file[ds] = segs[i]
        seg_file[ds].attrs["threshold"] = t

    return segs

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
    print(fg_indices)
    return tuple(
        slice(np.min(fg_indices[d]),np.max(fg_indices[d])+1)
        for d in range(3)
    )

def is_testing_sample(sample):

    if '+' in sample:
        return True
    return False

def create_segmentations(setup, iteration, sample, augmentation, seg_thresholds):

    if isinstance(setup, int):
        setup = 'setup%02d'%setup

    seg_thresholds = list(seg_thresholds)

    augmentation_name = sample
    if augmentation is not None:
        augmentation_name += '.augmented.' + str(augmentation)

    aff_data_dir = os.path.join(os.getcwd(), 'processed', setup, str(iteration))
    aff_filename = os.path.join(aff_data_dir, augmentation_name + ".hdf")

    print "Creating segmentation for " + sample + " with " + setup + ", iteration " + str(iteration) + " at thresholds " + str(seg_thresholds)

    print "Reading ground truth..."
    if is_testing_sample(sample):
        orig_data_dir = '../00_dataset_preparation/test/'
    else:
        with open(os.path.join('../02_train/', setup, 'train_options.json'), 'r') as f:
            orig_data_dir = json.load(f)['data_dir']
    orig_filename = os.path.join(orig_data_dir, augmentation_name + '.hdf')
    orig_file = h5py.File(orig_filename, 'r')
    gt = orig_file['volumes/labels/neuron_ids']

    print "Reading affinities..."
    aff_file = h5py.File(aff_filename, 'r')
    affs = aff_file['main']

    print "Getting ground-truth bounding box..."
    bb = get_gt_bounding_box(gt)

    print "Cropping affinities to ground-truth bounding box..."
    affs = crop(affs, bb)

    print "Copying affs to memory..."
    # for zwatershed...
    affs = np.array(affs)
    orig_file.close()
    aff_file.close()
    show_stats(affs, "affs")

    start = time.time()
    print "Getting segmentations for " + augmentation_name
    segs = create_watersheds(affs, seg_thresholds, aff_filename)
    print "Finished watershed in " + str(time.time() - start) + "s"

    print "sample    : " + sample
    print "setup     : " + setup
    print "thresholds: " + str(seg_thresholds)

    print "Storing meta-data"
    for i in range(len(seg_thresholds)):
        seg_file = h5py.File(seg_filename(aff_filename, seg_thresholds[i]), 'r+')
        seg_file['main'].attrs['orig_file'] = orig_filename
        seg_file['main'].attrs['offset'] = tuple(bb[d].start for d in range(3))
        seg_file.close()

if __name__ == "__main__":

    # NOTE: this is for debugging, normally create_segmentations is called by 
    # luigi
    create_segmentations('setup26', 100000, 'sample_B', '0', [50000])

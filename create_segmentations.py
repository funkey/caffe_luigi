from scipy import ndimage as ndi
import h5py
import json
import math
import numpy as np
import os
import sys
import time
import mahotas as mh
import waterz

def show_stats(v, name):
    print name + ":"
    print "\tshape :" + str(v.shape)
    print "\tdtype :" + str(v.dtype)

def seg_filename(aff_filename, threshold):
    threshold_string = ('%f'%threshold).rstrip('0').rstrip('.')
    return aff_filename[:-3] + threshold_string + ".hdf"

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

def create_boundary_thresholds(affs, seg_thresholds):

    # pixel-wise predictions = average of x and y affinity
    print "Computing membrane map"
    membrane = (affs[1,1:,1:,1:] + affs[2,1:,1:,1:])*0.5

    segs = []
    for t in seg_thresholds:

        segs.append(threshold_cc(membrane, t))

    return segs

def create_watersheds(affs, seg_thresholds, aff_filename, treat_as_boundary_map = False):

    print "Computing watersheds"
    if treat_as_boundary_map:
        segs = create_boundary_map_watersheds(affs, seg_thresholds)
    else:
        segs = waterz.agglomerate(affs, seg_thresholds)

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

def create_segmentations(setup, iteration, sample, augmentation, seg_thresholds, treat_as_boundary_map = False):

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
    # for waterz
    affs = np.array(affs)
    orig_file.close()
    aff_file.close()
    show_stats(affs, "affs")

    start = time.time()
    print "Getting segmentations for " + augmentation_name
    segs = create_watersheds(affs, seg_thresholds, aff_filename, treat_as_boundary_map)
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
    create_segmentations('setup26', 100000, 'sample_B', '0', [0, 0.1, 0.3, 0.5, 0.7], treat_as_boundary_map = True)

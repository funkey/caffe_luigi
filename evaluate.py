import cremi
import h5py
import json
import numpy as np
from scipy.ndimage import binary_erosion
import os
import time
import waterz
import sys
from agglomerate import agglomerate
from roi import Roi
from coordinate import Coordinate

# in voxel
ids_border_threshold = 10

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

def get_gt_roi(gt):

    # no-label ids are <0, i.e. the highest numbers in uint64
    fg_indices = np.where(gt <= np.uint64(-10))
    return Roi(
            (np.min(fg_indices[d]) for d in range(3)),
            (np.max(fg_indices[d])+1 - np.min(fg_indices[d]) for d in range(3))
    )

def evaluate(
        setup,
        iteration,
        sample,
        thresholds,
        output_basenames,
        custom_fragments,
        histogram_quantiles,
        discrete_queue,
        merge_function = None,
        section_wise=False,
        init_with_max = True,
        dilate_mask = 0,
        mask_fragments = False,
        keep_segmentation = False,
        aff_high = 0.9999,
        aff_low = 0.0001,
        *args,
        **kwargs):

    if isinstance(setup, int):
        setup = 'setup%02d'%setup

    thresholds = list(thresholds)

    aff_data_dir = os.path.join(os.getcwd(), 'processed', setup, str(iteration))
    affs_filename = os.path.join(aff_data_dir, sample + ".hdf")

    print "Evaluating " + sample + " with " + setup + ", iteration " + str(iteration) + " at thresholds " + str(thresholds)

    print "Reading ground-truth..."
    gt_filename = os.path.join('../01_data', sample + '.hdf')
    gt_file = cremi.io.CremiFile(gt_filename, 'r')
    gt_volume = gt_file.read_volume('volumes/labels/cells')

    resolution = Coordinate(gt_volume.resolution)

    # we have a "GT data ROI" of the whole GT volume, and a "GT ROI" in which 
    # tightly fits the non-background parts
    gt_data_offset_nm = Coordinate(gt_volume.offset)
    gt_data_roi = Roi(gt_data_offset_nm/resolution, gt_volume.data.shape)

    print "Getting ground-truth bounding box..."
    gt_roi = get_gt_roi(gt_volume.data)
    gt_roi = gt_roi.shift(gt_data_roi.get_offset())
    print "GT data ROI : " + str(gt_data_roi)
    print "GT ROI (>=0): " + str(gt_roi)

    print "Reading affinities..."
    affs_file = h5py.File(affs_filename, 'r')
    affs = affs_file['volumes/predicted_affs']
    affs_data_offset_nm = Coordinate(affs_file['volumes/predicted_affs'].attrs['offset'])
    affs_data_roi = Roi(affs_data_offset_nm/resolution, affs.shape[1:])
    print "affs data ROI: " + str(affs_data_roi)

    common_roi = gt_roi.intersect(affs_data_roi)
    common_roi_in_gt   = common_roi - gt_data_roi.get_offset()
    common_roi_in_affs = common_roi - affs_data_roi.get_offset()

    print "Common ROI of GT and affs is " + str(common_roi)
    print "Common ROI in GT: " + str(common_roi_in_gt)
    print "Common ROI in affs: " + str(common_roi_in_affs)

    print "Cropping ground-truth to common ROI"
    print "Previous shape: " + str(gt_volume.data.shape)
    gt = gt_volume.data[common_roi_in_gt.get_bounding_box()]
    gt_volume.data = gt
    gt_file.close()
    print "New shape: " + str(gt.shape)

    print "Cropping affinities to common ROI"
    affs = np.array(affs[(slice(None),) + common_roi_in_affs.get_bounding_box()], dtype=np.float32)
    affs_file.close()
    assert affs.shape[1:] == gt.shape

    print "Setting all GT special labels to -1 (will be 0 later)"
    no_gt = gt==np.uint64(-3)
    artifact_mask = gt==np.uint64(-1)
    gt[no_gt] = -1

    print "Growing ground-truth boundary..."
    print("GT min/max: " + str(gt.min()) + "/" + str(gt.max()))
    evaluate = cremi.evaluation.NeuronIds(gt_volume, border_threshold=ids_border_threshold)
    gt_with_borders = np.array(evaluate.gt, dtype=np.uint32)
    print("GT with border min/max: " + str(gt_with_borders.min()) + "/" + str(gt_with_borders.max()))

    if dilate_mask != 0:
        print "Dilating GT mask..."
        # in fact, we erode the no-GT mask
        no_gt = binary_erosion(no_gt, iterations=dilate_mask, border_value=True)

    print "Masking affinities outside ground-truth..."
    for d in range(3):
        affs[d][no_gt] = 0

    if section_wise:
        print "Setting affinties between sections to 0 (section-wise merging)"
        affs[0] = 0

    start = time.time()

    fragments_mask = None
    if mask_fragments:
        print "Masking fragments outside ground-truth..."
        fragments_mask = no_gt==False

    i = 0
    for seg_metric_history in agglomerate(
            affs,
            gt_with_borders,
            thresholds,
            custom_fragments=custom_fragments,
            histogram_quantiles=histogram_quantiles,
            discrete_queue=discrete_queue,
            merge_function=merge_function,
            init_with_max=init_with_max,
            fragments_mask=fragments_mask,
            aff_high=aff_high,
            aff_low=aff_low,
            return_merge_history=True):

        output_basename = output_basenames[i]

        if keep_segmentation:

            print "Storing segmentation..."
            f = h5py.File(output_basename + '.hdf', 'w')
            seg = seg_metric_history[0]

            ds = f.create_dataset('volumes/labels/cells', seg.shape, compression="gzip", dtype=np.uint64)
            ds[:] = seg
            ds.attrs['offset'] = common_roi.get_offset()*resolution
            ds.attrs['resolution'] = resolution

            ds = f.create_dataset('volumes/labels/gt_cells', gt_with_borders.shape, compression="gzip", dtype=np.uint64)
            ds[:] = gt_with_borders
            ds.attrs['offset'] = common_roi.get_offset()*resolution
            ds.attrs['resolution'] = resolution

            f.close()

        print "Storing record..."

        metrics = seg_metric_history[1]
        threshold = thresholds[i]
        i += 1

        record = {
            'setup': setup,
            'iteration': iteration,
            'sample': sample,
            'threshold': threshold,
            'merge_function': merge_function,
            'init_with_max': init_with_max,
            'aff_high': aff_high,
            'aff_low': aff_low,
            'dilate_mask': dilate_mask,
            'mask_fragments': mask_fragments,
            'custom_fragments': custom_fragments,
            'histogram_quantiles': histogram_quantiles,
            'discrete_queue': discrete_queue,
            'raw': { 'filename': gt_filename, 'dataset': 'volumes/raw' },
            'gt': { 'filename': gt_filename, 'dataset': 'volumes/labels/gt' },
            'affinities': { 'filename': affs_filename, 'dataset': 'main' },
            'voi_split': metrics['V_Info_split'],
            'voi_merge': metrics['V_Info_merge'],
            'rand_split': metrics['V_Rand_split'],
            'rand_merge': metrics['V_Rand_merge'],
            'gt_border_threshold': ids_border_threshold,
            'waterz_version': waterz.__version__,
        }
        with open(output_basename + '.json', 'w') as f:
            json.dump(record, f)

        print "Storing merge history..."

        history = seg_metric_history[2]
        with open(output_basename + '.merge_history.json', 'w') as f:
            json.dump(history, f)

    print "Finished waterz in " + str(time.time() - start) + "s"

if __name__ == "__main__":

    args_file = sys.argv[1]
    with open(args_file, 'r') as f:
        args = json.load(f)
    evaluate(**args)

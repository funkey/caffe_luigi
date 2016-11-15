import h5py
import cremi
import numpy as np
import pyted

# in nm, equivalent to CREMI metric
neuron_ids_border_threshold = 25

def evaluate(filename):

    file = h5py.File(filename, 'r+')
    orig_filename = file['main'].attrs['orig_file']

    print("Reading GT and SEG")

    if 'resolution' not in h5py.File(orig_filename, 'r')['volumes/labels/neuron_ids'].attrs:
        # this is lost in the alignment/augmentation
        h5py.File(orig_filename, 'r+')['volumes/labels/neuron_ids'].attrs['resolution'] = (40,4,4)
    truth = cremi.io.CremiFile(orig_filename, 'r')
    gt_volume = truth.read_neuron_ids()
    seg = file['main']
    offset = seg.attrs['offset']
    seg = np.array(seg, dtype=np.uint32)

    print("Cropping GT to SEG")

    gt_volume.data = gt_volume.data[
        offset[0]:offset[0]+seg.shape[0],
        offset[1]:offset[1]+seg.shape[1],
        offset[2]:offset[2]+seg.shape[2],
    ]

    print("Growing GT borders")

    gt_volume.data[gt_volume.data>=np.uint64(-10)] = -1
    print("GT min/max: " + str(gt_volume.data.min()) + "/" + str(gt_volume.data.max()))
    evaluate = cremi.evaluation.NeuronIds(gt_volume, border_threshold=neuron_ids_border_threshold)
    gt_with_borders = np.array(evaluate.gt, dtype=np.uint32)
    print("GT with border min/max: " + str(gt_with_borders.min()) + "/" + str(gt_with_borders.max()))
    print("SEG min/max: " + str(seg.min()) + "/" + str(seg.max()))

    print("Evaluating")

    ted = pyted.Ted()
    report = ted.create_report(gt_with_borders, seg)
    print(report)

    # delete previous metric attributes, if existing
    for k in file['main'].attrs:
        if k not in ['orig_file', 'offset']:
            del file['main'].attrs[k]
    for m in ['voi_split', 'voi_merge', 'rand_precision', 'rand_recall', 'adapted_rand_error']:
        file['main'].attrs[m] = report[m]
    file['main'].attrs['evaluated'] = True

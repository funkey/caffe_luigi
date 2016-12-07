from evaluate import evaluate

def create_segmentations(setup, iteration, sample, augmentation, seg_thresholds, output_files, treat_as_boundary_map = False, tag = None):
    evaluate(setup, iteration, sample, augmentation, seg_thresholds, output_files, treat_as_boundary_map = treat_as_boundary_map, tag = tag, keep_segmentation = True)

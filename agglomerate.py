import waterz
import config
from watershed import watershed

def agglomerate(
        affs,
        gt,
        thresholds,
        custom_fragments = False,
        histogram_quantiles = False,
        discrete_queue = False,
        merge_function = None):

    fragments = None
    if custom_fragments:
        fragments = watershed(affs, 'maxima_distance')

    if histogram_quantiles:
        merge_function += '_histograms'

    discretize_queue = 0
    if discrete_queue:
        discretize_queue = 256

    return waterz.agglomerate(
            affs,
            thresholds,
            gt,
            fragments=fragments,
            scoring_function=config.scoring_function[merge_function],
            discretize_queue=discretize_queue)

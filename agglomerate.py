import waterz
from ext import zwatershed
import config
from watershed import watershed

def zwatershed_thresholds(
        affs,
        thresholds,
        gt,
        aff_high,
        aff_low):

    for threshold in thresholds:

        # we have to run zwatershed threshold by threshold, otherwise we run 
        # out of memory, as each segmentation is stored
        seg_stats = zwatershed.zwatershed(
            affs,
            [threshold],
            gt,
            aff_high,
            aff_low)

        segs, stats = seg_stats
        seg = segs[0]
        stats = {
            'V_Rand_split': stats['V_Rand_split'][0],
            'V_Rand_merge': stats['V_Rand_merge'][0],
            'V_Info_split': stats['V_Info_split'][0],
            'V_Info_merge': stats['V_Info_merge'][0]
        }

        yield (seg, stats)


def agglomerate(
        affs,
        gt,
        thresholds,
        custom_fragments = False,
        histogram_quantiles = False,
        discrete_queue = False,
        merge_function = None,
        init_with_max = True,
        fragments_mask = None,
        aff_high = 0.9999,
        aff_low = 0.0001):

    if merge_function is not 'zwatershed':

        fragments = None
        if custom_fragments:
            fragments = watershed(affs, 'maxima_distance')
            if fragments_mask is not None:
                fragments[fragments_mask==False] = 0

        if init_with_max:
            merge_function += '_maxinit'
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
                discretize_queue=discretize_queue,
                aff_threshold_high=aff_high,
                aff_threshold_low=aff_low)

    else:

        return zwatershed_thresholds(
                affs,
                thresholds,
                gt,
                aff_high,
                aff_low)

import waterz

if waterz.__version__ == '0.6':
    scoring_function = {
            'median_aff': 'OneMinus<QuantileAffinity<AffinitiesType, 50>>',
            'median_aff_histograms': 'OneMinus<QuantileAffinity<AffinitiesType, 50, HistogramQuantileProviderSelect<256>::Value>>',
            '85_aff': 'OneMinus<QuantileAffinity<AffinitiesType, 85>>',
            '85_aff_histograms': 'OneMinus<QuantileAffinity<AffinitiesType, 85, HistogramQuantileProviderSelect<256>::Value>>',
            'max_10': 'OneMinus<MaxKAffinity<AffinitiesType, 10>>'
    }
else:
    scoring_function = {

            'mean_aff': 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
            'max_10': 'OneMinus<MeanMaxKAffinity<RegionGraphType, 10, ScoreValue>>',

            # quantile merge functions, initialized with max affinity
            '15_aff_maxinit': 'OneMinus<QuantileAffinity<RegionGraphType, 15, ScoreValue>>',
            '15_aff_maxinit_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 15, ScoreValue, 256>>',
            '25_aff_maxinit': 'OneMinus<QuantileAffinity<RegionGraphType, 25, ScoreValue>>',
            '25_aff_maxinit_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256>>',
            'median_aff_maxinit': 'OneMinus<QuantileAffinity<RegionGraphType, 50, ScoreValue>>',
            'median_aff_maxinit_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>',
            '75_aff_maxinit': 'OneMinus<QuantileAffinity<RegionGraphType, 75, ScoreValue>>',
            '75_aff_maxinit_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256>>',
            '85_aff_maxinit': 'OneMinus<QuantileAffinity<RegionGraphType, 85, ScoreValue>>',
            '85_aff_maxinit_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>>',

            # quantile merge functions, initialized with quantile
            '15_aff': 'OneMinus<QuantileAffinity<RegionGraphType, 15, ScoreValue, false>>',
            '15_aff_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 15, ScoreValue, 256, false>>',
            '25_aff': 'OneMinus<QuantileAffinity<RegionGraphType, 25, ScoreValue, false>>',
            '25_aff_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
            'median_aff': 'OneMinus<QuantileAffinity<RegionGraphType, 50, ScoreValue, false>>',
            'median_aff_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
            '75_aff': 'OneMinus<QuantileAffinity<RegionGraphType, 75, ScoreValue, false>>',
            '75_aff_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
            '85_aff': 'OneMinus<QuantileAffinity<RegionGraphType, 85, ScoreValue, false>>',
            '85_aff_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256, false>>',
    }

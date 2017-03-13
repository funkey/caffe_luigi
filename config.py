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
            'median_aff': 'OneMinus<QuantileAffinity<RegionGraphType, 50, ScoreValue>>',
            'median_aff_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>',
            '85_aff': 'OneMinus<QuantileAffinity<RegionGraphType, 85, ScoreValue>>',
            '85_aff_histograms': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>>',
            'max_10': 'OneMinus<MeanMaxKAffinity<RegionGraphType, 10, ScoreValue>>'
    }

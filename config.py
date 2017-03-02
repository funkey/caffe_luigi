scoring_function = {
        'median_aff': 'OneMinus<QuantileAffinity<AffinitiesType, 50>>',
        'median_aff_histograms': 'OneMinus<QuantileAffinity<AffinitiesType, 50, HistogramQuantileProviderSelect<256>::Value>>',
        '85_aff': 'OneMinus<QuantileAffinity<AffinitiesType, 85>>',
        '85_aff_histograms': 'OneMinus<QuantileAffinity<AffinitiesType, 85, HistogramQuantileProviderSelect<256>::Value>>',
        'max_10': 'OneMinus<MaxKAffinity<AffinitiesType, 10>>'
}


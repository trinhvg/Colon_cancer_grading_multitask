import csv
import json
import operator

import numpy as np
#import pandas

# Note: extract valid_accuracy during across-validation dold

nr_fold = 5
log_path = '/media/vtltrinh/Data1/COLON_MANUAL_PATCHES/v1/1010711/'

fold_stat = []
for fold_idx in range(0, nr_fold):
    stat_file = '%s/%02d/stats.json' % (log_path, fold_idx)
    with open(stat_file) as f:
        info = json.load(f)

    best_value = 0
    for epoch in info:
        epoch_stats = info[epoch]
        epoch_value = epoch_stats['valid-acc']
        if best_value > best_value:
            best_value = epoch_value
    fold_stat.append(best_value)

print(fold_stat)

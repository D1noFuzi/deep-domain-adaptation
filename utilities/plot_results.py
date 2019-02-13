import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas
from math import floor
from random import sample
import matplotlib.ticker as ticker


def create_combined_stats(folder, suffix='', adda=False, _cols=100, _range=10):
    file_list = ['run_.-tag-Train_source_acc.csv', 'run_.-tag-Train_target_acc.csv',
                 'run_eval-tag-source_class_acc.csv', 'run_eval-tag-target_class_acc.csv']
    # file_list = ['run_.-tag-Train_source_acc.csv']
    stats = dict()
    for _file in file_list:
        stats[_file] = np.empty([_range, _cols])
    for i in range(_range):
        for _file in file_list:
            run = 'run_' + str(i+1) + suffix
            if adda:
                run = run + '/adversarial_model'
            with open(os.path.join(folder, run, _file)) as f:
                data = pandas.read_csv(f)
                row = data.iloc[:, 2]
                row = row.values
                if len(row) > _cols:
                    steps = floor(len(row) / _cols)
                    row_ = row[0::steps]
                    random_index = sample(range(len(row_)-1), len(row_)-_cols)
                    row = np.delete(row_, random_index)
                stats[_file][i] = row
    return stats

def create_combined_stats_pkl(folder, _cols=100):
    key_list = ['source_loss_train', 'source_acc_train', 'source_acc_test',
                 'target_loss_train', 'target_acc_train', 'target_acc_test']
    stats = dict()
    for _file in key_list:
        stats[_file] = np.empty([10, _cols])
    for i in range(10):
        run = 'run_' + str(i + 1)
        with open(os.path.join(folder, run, 'stats.pkl'), 'rb') as f:
            results = pickle.load(f)
            for key in key_list:
                stats[key][i] = [x[1] for x in results[key]]
    return stats


_cols = 400
stats = create_combined_stats('../RevGrad/model_s2m', _cols=_cols, suffix='_new', _range=1)  # create_combined_stats('../ADDA/model_m2mm', adda=True, _cols=_cols)
x = range(_cols)
y = stats['run_eval-tag-target_class_acc.csv'].mean(0)  # stats['target_acc_test'].mean(0)  #
_max = stats['run_eval-tag-target_class_acc.csv'].max(0)  # stats['target_acc_test'].max(0)  #
_min = stats['run_eval-tag-target_class_acc.csv'].min(0)  # stats['target_acc_test'].min(0)  #
_std = stats['run_eval-tag-target_class_acc.csv'].std(0)  # stats['target_acc_test'].std(0)  #
print('Mean: ' + str(stats['run_eval-tag-target_class_acc.csv'][:, 20:].mean()))
print('Max: ' + str(stats['run_eval-tag-target_class_acc.csv'][:, 20:].max()))
print('Min: ' + str(stats['run_eval-tag-target_class_acc.csv'][:, 20:].min()))
print('Std: ' + str(stats['run_eval-tag-target_class_acc.csv'][:, 20:].std()))

fig, ax = plt.subplots(1, 1)
axes = plt.gca()
axes.set_ylim([0, 1])
axes.set_xlim([0, _cols+10])
ax.yaxis.set_major_locator(ticker.MultipleLocator(.1))
# plt.bar(x, error_max-error_min, width=0.3, bottom=error_min, color='black', linewidth=1)
# plt.plot(x, y)
# plt.plot(x, y, 'or')

# (_, caps, _) = plt.errorbar(x, y, yerr=[y - _min, _max - y], capsize=3, elinewidth=0.1, label='Max and min')
# for cap in caps:
#     cap.set_color('red')
#     cap.set_markeredgewidth(2)

# plt.fill_between(x, y - _std, y + _std, alpha=0.2, facecolor='grey', label='Std')
plt.plot(x, y, label='Target accuracy average', zorder=10)

y = stats['run_eval-tag-source_class_acc.csv'].mean(0)  # stats['source_acc_test'].mean(0)  #
_max = stats['run_eval-tag-source_class_acc.csv'].max(0)  # stats['source_acc_test'].max(0)  #
_min = stats['run_eval-tag-source_class_acc.csv'].min(0)  # stats['source_acc_test'].min(0)  #
_std = stats['run_eval-tag-source_class_acc.csv'].std(0)  # stats['source_acc_test'].std(0)  #
print('Mean: ' + str(stats['run_eval-tag-source_class_acc.csv'][:, 20:].mean()))
print('Max: ' + str(stats['run_eval-tag-source_class_acc.csv'][:, 20:].max()))
print('Min: ' + str(stats['run_eval-tag-source_class_acc.csv'][:, 20:].min()))
print('Std: ' + str(stats['run_eval-tag-source_class_acc.csv'][:, 20:].std()))

plt.plot(x, y, label='Source accuracy average', zorder=10)

plt.legend(loc=4)
plt.grid(True)
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.title('SVHN to MNIST: accuracy of evaluation set')

plt.show()


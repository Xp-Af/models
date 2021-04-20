#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import re
import csv

y_score = []
y_true = []

target_data = 'Af'

# Prepare dir
inferenced_csv_dir = '../../Results/'  + target_data + '/Likelihood/'
roc_images_dir = '../../Results/' + target_data + '/ROC/'
# Prepare dir if neccesarry
os.makedirs(roc_images_dir, exist_ok=True)

#Latest directory
files_path = glob.glob(inferenced_csv_dir + '*')
files_path_sorted = sorted(files_path, key=lambda f: os.stat(f).st_mtime, reverse=True)
most_recent_path= files_path_sorted[0]
#model_datetime = most_recent_path.split('/',5)[5].rsplit('.',1)[0]
df = pd.read_csv(most_recent_path)

#Define scores
y_val_score = df[df['Split'] == 'val']['1']
y_val_true = df[df['Split'] == 'val']['Label']

y_test_score = df[df['Split'] == 'test']['1']
y_test_true = df[df['Split'] == 'test']['Label']

# Calculate AUC
fpr_val, tpr_val, thresholds_val = metrics.roc_curve(y_val_true, y_val_score)
auc_val = metrics.auc(fpr_val, tpr_val)
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test_true, y_test_score)
auc_test = metrics.auc(fpr_test, tpr_test)
print('val: ' + '%.2f'%auc_val + ', ' + 'test: ' + '%.2f'%auc_test)

#Plot ROC
basename = os.path.splitext(os.path.basename(most_recent_path))[0]
#roc_save_name = roc_images_dir + 'roc_' + '%.2f'%auc_val + '_' + '%.2f'%auc_test + '_' + model_datetime + '.png'
roc_save_name = roc_images_dir + 'roc_' + '%.2f'%auc_val + '_' + '%.2f'%auc_test + '_' + basename  + '.png'

plt.plot(fpr_val, tpr_val, label='AUC_val = %.2f'%auc_val, marker='x')
plt.plot(fpr_test, tpr_test, label='AUC_test = %.2f'%auc_test, marker='o')
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC')
plt.grid()
plt.savefig(roc_save_name)

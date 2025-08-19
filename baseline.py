import numpy as np
import pickle
import pandas as pd
import json
import os
import itertools
import copy
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from yardsticks_GMM_train import RESULTS_DICO, classes_to_level, level_to_int, save_results_to_csv, evaluate, mean_absolute_difference, load_annotations

yardsticks = ['structure', 'lexicon','syntax', 'semantics']
results = copy.deepcopy(RESULTS_DICO)

df = pd.read_csv("probs_hybrid.csv", index_col="text_indice")
df = df.iloc[:len(df) // 2]
#df = df[~df.index.duplicated(keep='first')]
df['classe'] = df['prediction'].map(classes_to_level)

yardticks_filename="aggregated_yardstick_annotations.csv"
yardstick_annotations = pd.read_csv(yardticks_filename, index_col=0)

for yardstick in yardsticks:
    yardstick_annotations[yardstick] = yardstick_annotations[yardstick].map(classes_to_level)

#print(yardstick_annotations)

for yardstick in yardsticks:
    #print(yardstick_annotations[yardstick])
    predictions = df['classe'].loc[yardstick_annotations[yardstick].index]
    y_true = yardstick_annotations[yardstick]
    #print(predictions)
    evaluate(y_true.tolist(), predictions.tolist(), results, yardstick)

results['config'] = f"baseline_prediction_hybride"
filename = "./results/results_baseline_predictions.csv"
save_results_to_csv(results, filename=filename)

"""pred_indices = filtered_classes.index
true_indices = yardstick_annotations[yardstick].index
same_order = pred_indices.equals(true_indices)"""


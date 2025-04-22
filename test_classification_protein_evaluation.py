"""
Author: Zhiyuan Zhang
Date: Dec 2021
Email: cszyzhang@gmail.com
Website: https://wwww.zhiyuanzhang.net
"""

import os
import sys
import pandas as pd




from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    data = pd.read_csv("test_set_ground_truth.csv")
    print(data.head())
    filenames = data["anonymised_protein_id"]
    y_true = data["class_id"]

    data_run1 = pd.read_csv("test_set_2_run1.csv")
    y_pred = data_run1["predicted_label"]
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    # Print results
    print("RUN 1:")
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    data_run2 = pd.read_csv("test_set_2_run2.csv")
    y_pred = data_run2["predicted_label"]
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    # Print results
    print("RUN 2:")
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    #correct1 = prediction_run1.eq(labels).sum()
    #correct2 = prediction_run2.eq(labels).sum()
    #return correct1/len(labels), correct2/len(labels)







import time

if __name__ == '__main__':

    main()

    #print("Run 1:", acc1)
    #print("Run 2:", acc2)
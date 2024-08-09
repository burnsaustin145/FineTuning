import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import csv

from sklearn import preprocessing
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from l_features_util import extract_features


def evaluate_response(response, model, scaler):
    """
    evaluate the current example for truthful features.
    predict() <- extract_features()
    :param response: string of current example
    :param model: svm model to use
    :param scaler: use a data scaler fitted to the same data model was trained on
    """
    # train data shape (164, 220)
    svm_model = model
    response_features = extract_features(response, features=None)
    response_features = list(response_features.values())

    response_arr = np.array(response_features)
    response_arr = response_arr.reshape(1, -1)
    response_arr = scaler.transform(response_arr)

    prediction = svm_model.predict(response_arr)
    if prediction == 1:
        print("Got a positive value!!!!!!!!!!!!!!!!!!!!!")
    return prediction, response_arr


def evaluate_tuning_examples(data_path, curr_example):
    """
    Uses evaluate_response to evaluate a tuning
    :param data_path: path to the current tuning example
    :param curr_example: name of the jsonl
    """
    model = pickle.load(open('svm_model_davinci.pkl', 'rb'))
    scalar = pickle.load(open('./data/scaler.pkl', 'rb'))
    with open(data_path, 'r') as file:
        data_list = list(file)
    with open(f'./data/svm_evaluated_tuning_examples/features_{curr_example}.csv', 'w') as file:
        with open(f'./data/svm_evaluated_tuning_examples/{curr_example}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer_features = csv.writer(file)
            for entry in data_list:
                curr = json.loads(entry)
                curr_response = curr['messages'][1]['content']
                curr_result, curr_features = evaluate_response(curr_response, model, scalar)
                writer.writerow([curr_response, curr_result])
                writer_features.writerow(curr_features)
    return None


def evaluate_features(features_path):
    model = pickle.load(open('svm_model_davinci.pkl', 'rb'))
    scalar = pickle.load(open('./data/scaler.pkl', 'rb'))
    with open(features_path, 'r') as file:
        data = csv.reader(file, delimiter=',')
        skip = 0
        for entry in data:
            if skip == 0:
                skip += 1
                continue
            entry.pop()
            curr = np.array(entry)
            curr = curr.reshape(1, -1)
            curr = scalar.transform(curr)
            curr_result = model.predict(curr)
            print(curr_result[0])
            if curr_result[0] == 1:
                print("Got a positive value!!!!!!!!!!!!!!!!!!!!!")

if __name__ == "__main__":
    # svm evaluation basic usage
    # pred = evaluate_response("This is just one example, probably false wouldn't you think.")
    # print(pred)
    """
    model = pickle.load(open('svm_model_davinci.pkl', 'rb'))
    scalar = pickle.load(open('./data/scaler.pkl', 'rb'))
    pred, feat = evaluate_response("just nothing rn I'm gettin mad yo", model, scalar)
    print(type(pred))
    print(pred)
    print(feat)
    """
    res = evaluate_features("./svm/truthfulqa_experiments/outputs/davinci_output_labeled_features.csv")

    """
    tuning_examples = ['fictional_truth_QA_tuning_.jsonl', 'truthful_QA_tuning.jsonl']
    for ex in tuning_examples:
        evaluate_dataset(f'./data/tuning_examples/{ex}', ex)
    """
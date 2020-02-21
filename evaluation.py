#!/usr/bin/env python3
# coding: utf-8
"""
title: eval.py
date: 2019-11-23
author: jskrable
description: Model evaluation, plotting, and user testing.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.ml.classification import NaiveBayesModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics


def important_words(model, n):
    """
    Function that finds the n most important words for a model
    """
    theta = np.array(model.theta.toArray())
    dictionary = np.array(CountVectorizerModel.load('./cv_model').vocabulary)
    ind = np.argpartition(theta, -n)[-n:]
    return dictionary[ind]


def binary_eval(df, label='indexedLabel', pred='prediction'):
    """
    Function to evaluate classification model. Takes in a df,
    the label column, and the prediction column.
    Returns the area under ROC.
    """
    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol=pred,
        labelCol=label)
    score = evaluator.evaluate(df)
    return score


def binary_metrics(rdd):
    """
    Function to evaluate classification model. Takes in a df,
    the label column, and the prediction column.
    Returns all available metrics.    
    """
    metrics = BinaryClassificationMetrics(rdd)
    # score = metrics.evaluate(rdd)
    return metrics


def plot_results(results):
    """
    """
    data = pd.DataFrame(
        {'score': [results[r]['score'] for r in results.keys()],
        'train': [results[r]['train_time'] for r in results.keys()],
        'label': [r for r in results.keys()]
        })
    ax = sns.set_style('darkgrid')
    ax = sns.scatterplot(x='score', y='train', data=data)
    for line in range(0,data.shape[0]):
        ax.text(data.score[line], 
            data.train[line], 
            data.label[line], 
            horizontalalignment='right', 
            size='small', 
            color='black')

    # ax.set(yticklabels=[])
    # ax.set(xticklabels=[])
    plt.show()


def predict_input(headline, model):
    data = [{'headline': headline}]
    print('preprocessing user input headline')
    df = preprocessing(data, False)
    print('making prediction')
    pred = model.transform(df).take(1)[0].prediction
    return bool(pred)


def take_input(model):
    user_headline = ''
    print('starting user input')
    while user_headline != 'quit':
        user_headline = input('Type in a headline: ')
        print(user_headline)
        # if user_headline in end_words:
        #     print('checking for end')
        #     break
        print('sending to prediction function')
        pred = predict_input(user_headline, model)
        print('Satire:   {}'.format(pred))
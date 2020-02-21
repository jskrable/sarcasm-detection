#!/usr/bin/env python3
# coding: utf-8
"""
title: main.py
date: 2019-11-23
author: jskrable
description: Model initialization and training.
"""

from timeit import default_timer as timer
from pyspark.ml.classification import RandomForestClassifier, NaiveBayes, NaiveBayesModel, GBTClassifier, LinearSVC


def random_forest(df, label, features):
    """
    Function to train a random forest classifier from an input df
    and return a trained classification model
    """
    print('Training random forest model...')
    rf = RandomForestClassifier(
        labelCol=label,
        featuresCol=features,
        numTrees=100,
        maxDepth=30)
    model = rf.fit(df)
    return model


def gradient_boosted_tree(df, label, features):
    """
    Function to train a gradient-boosted decision tree classifier 
    from an input df and return a trained classification model
    """
    print('Training gradient boosted tree model...')
    gbt = GBTClassifier(
        labelCol=label,
        featuresCol=features,
        maxIter=10)
    model = gbt.fit(df)
    return model


def linear_SVC(df, label, features):
    """
    Function to train a linear support vector machine from an input df
    and return a trained classification model
    """
    print('Training support vector machine model...')
    gbt = LinearSVC(
        labelCol=label,
        featuresCol=features,
        maxIter=100,
        regParam=0.1)
    model = gbt.fit(df)
    return model


def naive_bayes(df, label, features, save=False):
    """
    Function to train a naive bayes classifier from an input df
    and return a trained classification model
    """
    print('Training naive bayes model...')
    nb = NaiveBayes(
        smoothing=1.0,
        modelType='multinomial',
        labelCol=label,
        featuresCol=features)
    model = nb.fit(df)
    if save:
        model.save('./nb_model')
    return model


def train_model(df, algorithm, label='indexedLabel', features='vector'):
    """
    generic wrapper function to train a classification model.
    takes in a df, the algorithm to use, the label column, and the features column.
    returns a trained model and the training time in seconds. 
    """
    s_train = timer()
    f = globals()[algorithm]
    model = f(df, label, features)
    e_train = timer()
    result = {'model': model, 'train_time': (e_train - s_train)}
    return result

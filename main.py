#!/usr/bin/env python3
# coding: utf-8
"""
title: main.py
date: 2019-11-23
author: jskrable
description: Term project for CS777. Sarcasm detection in headlines.
"""

import sys
import train as tr
import prep as pp
import evaluation as ev
from timeit import default_timer as timer
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: main.py <size> <basedir> <output> ", file=sys.stderr)
        exit(-1)

    basedir = sys.argv[1]
    size = int(sys.argv[2])
    output = sys.argv[3]

    sc = SparkContext(appName="Term Project: Sarcasm Detection")
    sc.setLogLevel("ERROR")
    sql = SQLContext(sc)

    print('\nPRE-PROCESSING-----------------------------------------------------------')
    start = timer()
    s_prep = timer()
    print('Reading dataset...')
    # data = pp.get_source_data('./data')[:25000]
    # data = pp.get_source_data(basedir)[:size]
    data = pp.get_source_data(basedir)
    print('Total observations: {}'.format(len(data)))
    print('Sending to pre-processing...')
    train_df, test_df = pp.preprocessing(sql, data)
    e_prep = timer()
    
    print('\nTRAINING-----------------------------------------------------------------')
    s_train = timer()
    models = ['naive_bayes','random_forest','linear_SVC','gradient_boosted_tree']
    results = {m: tr.train_model(train_df, m) for m in models}
    e_train = timer()

    print('\nTESTING------------------------------------------------------------------')
    s_test = timer()
    print('Making predictions...')
    for m in models:
        preds = results[m]['model'].transform(test_df)
        score = ev.binary_eval(preds)
        metrics = ev.binary_metrics(
            preds.select(['prediction','indexedLabel']).rdd.map(lambda x: (x[0],x[1]))
            )
        results[m].update({'score': score, 'metrics': metrics})
    [print(m + ' model accuracy  : {:2.6f}'.format(results[m]['score'])) for m in models]
    [print(m + ' train time      : {:2.6f}'.format(results[m]['train_time'])) for m in models]
    e_test = timer()

    ev.plot_results(results)

    end = timer()
    print('\nTIMING-------------------------------------------------------------------')
    print('Input    : {} seconds'.format(s_prep-start))
    print('Prep     : {} seconds'.format(e_prep-s_prep))
    print('Training : {} seconds'.format(e_train-s_train))
    print('Testing  : {} seconds'.format(e_test-s_test))
    print('----------------------------------------------')
    print('Total    : {} seconds'.format(end-start))

    # print('\nTry it out!')
    # take_input(nb_model)
    # print('\nSaving model to ')
    # rf_model.save(sys.argv[1])
    sc.stop()

#!/usr/bin/env python3
# coding: utf-8
"""
title: main.py
date: 2019-11-23
author: jskrable
description: Preprocessing
"""

import os
import re
import json
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import lit, when, col, udf, concat
from pyspark.ml.feature import NGram, CountVectorizer, CountVectorizerModel, IndexToString, StringIndexer, VectorIndexer


def progress(count, total, suffix=''):
    """
    Progress bar for cli
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


def parse_data(file):
    """
    Function to read a psuedo-json file line by line and
    return a generator object to save CPU and mem.
    Wrap response in a list() to subscribe
    """
    for l in open(file, 'r'):
        yield json.loads(l)


def get_source_data(base_dir):
    """
    Function to read a directory full of psuedo-json files 
    and return a list of objects. Objects are structured as follows:
    article_link: http link to original article
    headline: string headline, special characters intact
    is_sarcastic: int, 1 for sarcasm, 0 for serious
    """
    for d, _, f in os.walk(base_dir):
        files = [os.path.join(d,file) for file in f]
    data = [list(parse_data(f)) for f in files]
    data = [item for sublist in data for item in sublist]
    return data


def count_vectorizer(df, col, train=False):
    """
    Function to take in a df of headlines and tranform to a
    word count vector. Simple bag of words method. 
    Requires headline to be a list of words. Returns a df
    with an additional vector column.
    """
    if train:
        cv = CountVectorizer(
            inputCol=col,
            outputCol='vector',
            vocabSize=50000)
        model = cv.fit(df)
        print('Saving count vectorizer model to disk...')
        model.save('./cv_model')
    else:
        model = CountVectorizerModel.load('./cv_model')
    df = model.transform(df)
    return df


def label_indexer(df, col):
    """
    Function to take in a df, index the class label column,
    and return the df w/ a new indexedLabel column.
    """
    labelIndexer = StringIndexer(
        inputCol=col, 
        outputCol="indexedLabel").fit(df)
    df = labelIndexer.transform(df)
    return df


def n_grams(df, col, n=2):
    """
    Function to take in a df with a list of words and convert to
    list of n-grams.
    """
    ngram = NGram(
        n=n,
        inputCol=col, 
        outputCol="ngrams")
    df = ngram.transform(df)
    return df


def preprocessing(sql, data, train=True):
    """
    Function to take in a list of dicts containing string 
    headlines and return a df containing indexed labels and 
    vectorized features.
    """
    # convert input data to spark dataframe
    # print('Creating dataframe...')
    df = sql.createDataFrame(Row(**entry) for entry in data)
    # print('Cleaning headlines...')
    # allow only alphabetic characters
    regex = re.compile('[^a-zA-Z]')
    clean_headline = udf(lambda x: 
        regex.sub(' ', x).lower().split(), ArrayType(StringType()))
    df = df.withColumn('cleanHeadline', clean_headline(df.headline))
    df = n_grams(df, 'cleanHeadline')
    concat = udf(lambda x,y : x + y, ArrayType(StringType()))
    df = df.withColumn('gramList', concat(df.cleanHeadline,df.ngrams))
    # print('Vectorizing headlines...')
    # get a sparse vector of dictionary word counts
    # choose to use n-grams or list here
    # df = count_vectorizer(df, 'cleanHeadline')
    # df = count_vectorizer(df, 'ngrams')
    df = count_vectorizer(df, 'gramList', train)
    if train:
        # index label column
        print('Indexing labels...')
        df = label_indexer(df, 'is_sarcastic')
        train, test = df.randomSplit([0.7,0.3])
        return train, test
    else:
        return df
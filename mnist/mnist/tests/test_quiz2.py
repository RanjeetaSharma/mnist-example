from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump, load
from skimage import data, color
from skimage.transform import rescale
import numpy as np
import pickle
import os
from util import pre_processing,split_data,metric_test,run_classification_exp
import math


def test_create_split():
    digits_data = datasets.load_digits()
    x, y = digits_data.data, digits_data.target
    x_train, x_test, x_val, y_train, y_test, y_val = split_data(x, y, 0.3,0.1)
    train_data = len(x_train)
    test_data = len(x_test)
    val_data = len(x_val)
    data_len = len(x)
    assert train_data == math.trunc(data_len*0.7)



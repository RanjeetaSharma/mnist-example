import math
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump, load
from skimage import data, color
from skimage.transform import rescale
import numpy as np
import pickle
import os
from util import pre_processing,split_data,metric_test,run_classification_exp

def test1_sqrt():
    num = 25
    assert math.sqrt(num)==5


def test_square():
    num = 7
    assert 7*7 == 49

def test_equality():
    assert 10 == 10

def test_model_writing():
    gamma = 0.01
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    rescale_imgs = pre_processing(digits.images, 10)
    rescale_imgs = np.array(rescale_imgs)
    X_train,X_test,X_val,y_train,y_test,y_val = split_data(data, digits.target,0.3,0.1)
    clf = svm.SVC(gamma=gamma)
    clf.fit(X_train, y_train)

    metrics_valid = run_classification_exp(clf, X_train, y_train, X_val, y_val, gamma, '/home/ranjeeta/miniconda3/mnist-example/mnist/mnist/model/finalized_model')
    assert os.path.isfile('/home/ranjeeta/miniconda3/mnist-example/mnist/mnist/model/finalized_model')

def test_small_data_overfit_checking():
    gamma = 0.001
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data =digits.images.reshape((n_samples, -1))
    rescale_imgs = pre_processing(digits.images, 10)
    rescale_imgs = np.array(rescale_imgs)
    X = digits.data
    y = digits.target
    X= X[:100]
    y = y[:100]
    X_train,X_test,X_val,y_train,y_test,y_val = split_data(X, y, 0.2,0.1)
    clf = svm.SVC(gamma=gamma)
    clf.fit(X_train, y_train)
    train_metrics  = run_classification_exp(clf, X_train, y_train, X_train, y_train, gamma, '/home/ranjeeta/miniconda3/mnist-example/mnist/mnist/model/finalized_model')
    assert train_metrics["Accuracy"] > 0.50
    assert train_metrics["f1_score"] > 0.50
    



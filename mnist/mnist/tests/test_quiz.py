from flask import Flask, request
import math
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump, load
from skimage import data, color
from skimage.transform import rescale
import numpy as np
import pickle
import os
#from util import pre_processing,split_data,metric_test,run_classification_exp,load
from util import util
app = Flask(__name__)
## Loading the best SVM Model

clf = util.load('/home/ranjeeta/miniconda3/mnist-example/mnist/mnist/model/finalized_model')

## Loading the best DT Model

clf1 = util.load('/home/ranjeeta/miniconda3/mnist-example/mnist/mnist/model/finalized_model_DT')

def test_model_svm():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    rescale_imgs = pre_processing(digits.images, 10)
    rescale_imgs = np.array(rescale_imgs)
    X_train,X_test,X_val,y_train,y_test,y_val = split_data(data, digits.target,0.3,0.1)
    clf.fit(X_train, y_train)

def test_model_DT():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    rescale_imgs = pre_processing(digits.images, 10)
    rescale_imgs = np.array(rescale_imgs)
    X_train,X_test,X_val,y_train,y_test,y_val = split_data(data, digits.target,0.3,0.1)
    clf1.fit(X_train, y_train)

def test_digit_correct_0():
    prediction = clf.predict(X_test)
    assert prediction==0:

def test_digit_correct_svm_0():
    pred_svm_1 = clf1.predict(X_test)
    assert pred_svm_1==0:
def test_digit_correct_svm_1():
    pred_svm_2 = clf1.predict(X_test)
    assert pred_svm_2==0:

def test_digit_correct_svm_2():
    pred_svm_3 = clf1.predict(X_test)
    assert pred_svm_3==0:

def test_digit_correct_svm_3():
    pred_svm_31 = clf1.predict(X_test)
    assert pred_svm_31==0:

def test_digit_correct_svm_4():
    pred_svm_4 = clf1.predict(X_test)
    assert pred_svm_4==0:

def test_digit_correct_svm_5():
    pred_svm_5 = clf1.predict(X_test)
    assert pred_svm_5==0:

def test_digit_correct_svm_6():
    pred_svm_6 = clf1.predict(X_test)
    assert pred_svm_6==0:

def test_digit_correct_svm_7():
    pred_svm_7 = clf1.predict(X_test)
    assert pred_svm_7==0:

def test_digit_correct_svm_8():
    pred_svm_8 = clf1.predict(X_test)
    assert pred_svm_8==0:

def test_digit_correct_dt_2():
    pred_svm_dt_2 = clf.predict(X_test)
    assert pred_svm_dt_2==0:

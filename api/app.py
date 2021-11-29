from flask import Flask, request
import os 
import sys
import numpy as np

#Find the Util package
testdir = os.getcwd()
sys.path.insert(0, "/".join(testdir.split("/")[:-1] + ["mnist"]))

from util import load


app = Flask(__name__)
SVM_clf = load('/home/ranjeeta/miniconda3/mnist-example/mnist/mnist/model/svm_best_model.pkl')

DT_clf = load('/home/ranjeeta/miniconda3/mnist-example/mnist/mnist/model/dtree_best_model.pkl')


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict_svm", methods=['POST', 'GET'])
def svm_pred():
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1, -1)
    predict = SVM_clf.predict(image)
    return str(predict[0])

@app.route("/predict_dt", methods=['POST', 'GET'])
def dt_pred():
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1, -1)
    predict = DT_clf.predict(image)
    return str(predict[0])

app.run('0.0.0.0', debug = True, port = '5000')

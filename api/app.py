from flask import Flask, request
import os 
import sys
import numpy as np
from util import util
#To find the utils.utils package
testdir = os.getcwd()
sys.path.insert(0, "/".join(testdir.split("/")[:-1] + ["mnist"]))



app = Flask(__name__)
clf = util.load('/home/ranjeeta/miniconda3/mnist-example/mnist/mnist/model/finalized_model')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods=['POST', 'GET'])
def predict():
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    return str(predicted[0])

app.run('0.0.0.0', debug = True, port = '5000')

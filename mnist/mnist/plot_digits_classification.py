"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump, load
from skimage import data, color
from skimage.transform import rescale
import numpy as np
import pickle
import os
from util import pre_processing,split_data,test


###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
print("size of the Image is:")
print(digits.images[0].shape)
rescale_val = [1]
gamma_val= [1, 0.1, 0.01, 0.001, 10]
best_gamma = 0
best_accuracy = 0
best_f1_score = 0
print("Gamma Value\tTest Accuracy\t\tTest F1 Score\t\tValidation Accuracy\tValidation F1 Score")
for rescale_factor in rescale_val:
  for val in gamma_val:
    rescale_imgs = pre_processing(digits.images, rescale_factor)
    rescale_imgs = np.array(rescale_imgs)
    data = rescale_imgs.reshape((n_samples, -1))
    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=val)
    X_train,X_test,X_val,y_train,y_test,y_val = split_data(data, digits.target,0.3,0.1)
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    test_metrics = test(clf,X_test,y_test)
    val_metrics = test(clf,X_val,y_val)
    print("{}\t\t{}\t{}\t{}\t{}".format(val,test_metrics['Accuracy'],test_metrics['f1_score'],val_metrics['Accuracy'],val_metrics['f1_score']))
    # print(test_metrics)
    # best_gamma.append(val)
    # best_accuracy.append(acc)
    #Discard the models that yield random-like performance

    if val_metrics['Accuracy'] < 0.11:
       print("Discard for {}".format(val))
       continue
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(clf, open('/home/ranjeeta/miniconda3/mnist-example/mnist/mnist/model/finalized_model','wb')
            )
    print('Model is saved into to disk successfully Using Pickle')
    if val_metrics['Accuracy'] > best_accuracy:
       best_gamma = val
       best_accuracy = val_metrics['Accuracy']
       best_f1_score = val_metrics['f1_score']

    # Load Model from the disk
    my_model = pickle.load(open('/home/ranjeeta/miniconda3/mnist-example/mnist/mnist/model/finalized_model','rb'))
    result = my_model.predict(X_val)
    acc = metrics.accuracy_score(y_pred=result, y_true=y_val)
    f1 = metrics.f1_score(y_pred=result, y_true=y_val, average="macro")
    print("Validation Accuracy using saved Model",acc)
    print("Validation F1 Score using saved Model",f1)

print("The best gamma value is:",best_gamma)     
print("The best Validation accuracy value is:",best_accuracy)  
print("The best Validation F1 Score value is:",best_f1_score)  




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
from sklearn import datasets, svm, metrics, tree
from sklearn.model_selection import train_test_split,GridSearchCV


from skimage import data, color
from skimage.transform import rescale
import numpy as np

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
print("Image size is:")
print(digits.images[0].shape)


rescale_val = [1, 2, 3]
test_split = [0.2, 0.3, 0.4, 0.5, 0.6]


print("Train-Test Split  Accuracy (DT)\t\t Accuracy (SVM)")
for test in test_split:
        # data = digits
        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_split=10, min_samples_leaf=5)
        #clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_split=10, min_samples_leaf=5)
        clf_svm = svm.SVC(gamma = 0.001)

        # Split data into train and test subsets
        X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=test, shuffle=False)

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)
        clf_svm.fit(X_train, y_train)
        # Predict the value of the digit on the test subset
        predicted = clf.predict(X_test)
        predicted_smv = clf_svm.predict(X_test)

        acc_dt = metrics.accuracy_score(y_pred=predicted, y_true=y_test)
        acc_svm = metrics.accuracy_score(y_pred=predicted_smv, y_true=y_test)
        # f1 = metrics.f1_score(y_pred=predicted, y_true=y_test, average='macro')
        print("{}:{}\t{}\t{}".format((1-test)*100,test*100,acc_dt*100,acc_svm*100))


print("Mean of Decision Tree Classsifier:",acc_dt.mean()*100)
print("Mean of SVM Classsifier:",acc_svm.mean()*100)
print("Standard Deviation of Decision Tree Classsifier:",acc_dt.std())
print("Standard Deviation of SVM Classsifier:",acc_svm.std())

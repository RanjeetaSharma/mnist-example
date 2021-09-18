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

resize_imgs=[]
rescale_val = [1, 2, 3]
test_split = [0.2, 0.3, 0.4]
gamma_val= [1, 0.1, 0.01, 0.001, 10]

print("Image Size  Train-Test Split  Gamma val  Accuracy\t\t  F1 Score")
for val in rescale_val:
    for img  in digits.images:
        resize_imgs.append(rescale(img,val, anti_aliasing=False))
    for test in test_split:
        resize_imgs = np.array(resize_imgs)
        data = resize_imgs.reshape((n_samples, -1))
        for g in gamma_val:
            # Create a classifier: a support vector classifier
            clf = svm.SVC(gamma= g)

            # Split data into train and test subsets
            X_train, X_test, y_train, y_test = train_test_split(
                data, digits.target, test_size=test, shuffle=False)

            # Learn the digits on the train subset
            clf.fit(X_train, y_train)

            # Predict the value of the digit on the test subset
            predicted = clf.predict(X_test)

            ###############################################################################
            # Below we visualize the first 4 test samples and show their predicted
            # digit value in the title.

            _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
            for ax, image, prediction in zip(axes, X_test, predicted):
                ax.set_axis_off()
                image = image.reshape(resize_imgs[0].shape)
                ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
                ax.set_title(f'Prediction: {prediction}')

            ###############################################################################
            # :func:`~sklearn.metrics.classification_report` builds a text report showing
            # the main classification metrics.

            # print(f"Classification report for classifier {clf}:\n"
            #     f"{metrics.classification_report(y_test, predicted)}\n")

            ###############################################################################
            # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
            # true digit values and the predicted digit values.

            disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
            disp.figure_.suptitle("Confusion Matrix")
            # print(f"Confusion matrix:\n{disp.confusion_matrix}")

            acc = metrics.accuracy_score(y_pred=predicted, y_true=y_test)
            f1 = metrics.f1_score(y_pred=predicted, y_true=y_test, average='macro')
            print("{}x{}\t\t{}:{}\t{}\t{}\t{}".format(resize_imgs[0].shape[0], resize_imgs[0].shape[1], (1-test)*100,test*100,g,acc*100,f1*100))

            plt.show()

## Finding the best estimatior
param_grid = { 'gamma': [1, 0.1, 0.01, 0.001, 10]}
# Make grid search classifier

clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
# Train the classifier

clf_grid.fit(X_train, y_train)
# clf = grid.best_estimator_()

print("Best Estimators:\n", clf_grid.best_estimator_)

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

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

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
data      = digits.images.reshape((n_samples, -1))

def split_data(data, target):
  X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
  
  X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

  return X_train, y_train, X_test, y_test, X_val, y_val

def train_split(X_train, y_train, test_size):
  train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, test_size = test_size, random_state=42)
  return train_X, train_y

X_train, y_train, X_test, y_test, X_val, y_val = split_data(data, digits.target)

samples_train  = []
pred   = []
train_accuracy = []
test_accuracy  = []
val_accuracy = []
f1_score_metric  = []


for i in range (10):
  test_size = 1 - ((i + 1) / 10)

  if i == 9:
    train_X = X_train
    train_y = y_train
  else:
    train_X, train_y = train_split(X_train, y_train, test_size = test_size)

  samples_train.append(((i + 1) / 10) * 100)

  clf = tree.DecisionTreeClassifier()
  clf.fit(train_X, train_y)

  prediction      = clf.predict(X_val)
  acc_val = metrics.accuracy_score(y_pred = prediction, y_true = y_val)
  val_accuracy.append(acc_val)

  train_pred  = clf.predict(train_X)
  train_acc = metrics.accuracy_score(y_pred = train_pred, y_true = train_y)
  train_accuracy.append(train_acc)

  test_pred  = clf.predict(X_test)

  test_acc = metrics.accuracy_score(y_pred = test_pred, y_true = y_test)
  f1_score      = metrics.f1_score(y_test, test_pred, average = 'macro')

  test_accuracy.append(test_acc)
  f1_score_metric.append(f1_score)
  pred.append(test_pred)

x_axis = samples_train
y_axis = f1_score_metric

plt.figure(figsize=(8, 6))
plt.title('Training Samples v/s F1 Score')
plt.xlabel('X axis - Training Samples')
plt.ylabel('Y axis - f1 Score')
plt.xlim(0, 100)
plt.plot(x_axis, y_axis, color = 'blue')
plt.show()


print ("{:<15} {:<17} {:<20} {:<17} {:18}".format('Run', 'Train', 'Test', 'Val', 'F1 Score'))
print('=============================================================================================================================')

for i in range (10):
  print ((i + 1), '\t \t', round(train_accuracy[i]*100,2),'\t \t', round(test_accuracy[i]*100,2),'\t\t', round(val_accuracy[i]*100,2),'\t\t', round(f1_score_metric[i]*100,2))


print('Confusion Matrix')
print ('\n')

print('Training Samples: 30%')
confusion_mat = metrics.confusion_matrix(y_test, pred[2])
sn.heatmap(confusion_mat, annot=True)
print ('\n')

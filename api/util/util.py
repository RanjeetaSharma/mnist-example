from skimage.transform import rescale
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import os
import pickle

def pre_processing(imgs, rescale_factor):
  rescale_imgs =[]
  for img in imgs:
      rescale_imgs.append(rescale(img,rescale_factor, anti_aliasing=False))
  return rescale_imgs


## Define a function to split train test set

def split_data(data, target,test_size,val_size):

  # Split data into 70% train and 30% test subsets
  X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=test_size, shuffle=False)
  
  # Split data into 20% validation set and 10% test set
  X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=val_size, shuffle=False)
  return X_train,X_test,X_val,y_train,y_test,y_val

## Define Test function to calculate metrics on test set

def test(clf,X_test,y_test):
  # Predict the value of the digit on the test subset
    predicted_test = clf.predict(X_test)
    acc_test = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)
    f1_test = metrics.f1_score(y_pred=predicted_test, y_true=y_test, average='macro')

    return {'Accuracy':acc_test,'f1_score':f1_test}
def metric_test(clf,X_test,y_test):
  # Predict the value of the digit on the test subset
    predicted_test = clf.predict(X_test)
    acc_test = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)
    f1_test = metrics.f1_score(y_pred=predicted_test, y_true=y_test, average='macro')

    return {'Accuracy':acc_test,'f1_score':f1_test}

def run_classification_exp(clf, X_train, y_train, X_val, y_val, gamma, filename):
    random_acc = max(np.bincount(y_val)) / len(y_val)
    ## create a svm classifier
    clf = svm.SVC(gamma=gamma)
    clf.fit(X_train, y_train)
    #Predict values on val subset
    valid_metrics = metric_test(clf, X_val, y_val)
    if valid_metrics["Accuracy"] < random_acc:
        print("Ski for {}".format(gamma))
        return None

    out_folder = os.path.dirname(filename)
    pickle.dump(clf, open(filename,'wb'))
    return valid_metrics

def load(model_path):
     return pickle.load(open(model_path, 'rb'))

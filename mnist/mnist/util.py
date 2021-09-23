from skimage.transform import rescale
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics

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

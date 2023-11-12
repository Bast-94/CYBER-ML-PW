import numpy as np
# Test fgsm_attack_svm_2c
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# import makeblobs
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
def fgsm_attack_svm_2c(classifier, orig_point, dist_function, step=None, epsilon=np.inf, max_step=200):
    data_point = orig_point.copy()
    orig_class = classifier.predict(data_point.reshape(1, -1))[0]
    new_class = orig_class
    current_eps = dist_function(data_point,orig_point)
    attack_info = {}
    
    
    if step is None:
        step = 0.01
    i =0
    
    eps_evol = [current_eps]
    while orig_class == new_class:
        if current_eps < epsilon and i < max_step:
            # get datapoint by gradient descent
            grad = classifier.coef_[0]
            data_point = data_point + step * np.sign(grad) 
                        
            new_class = classifier.predict(data_point.reshape(1, -1))[0]
            current_eps = dist_function(data_point,orig_point)
            attack_info = dict(data_point=data_point, current_eps=current_eps)
            eps_evol.append(current_eps)
        else:
            attack_info = dict(data_point=None, current_eps=None)
            
            if current_eps > epsilon:
                print("Attack failed: epsilon exceeded")
            if step >= max_step:
                print("Attack failed: max step exceeded")
            print("Step:", i)
            
            break
        i += step
    fig, ax = plt.subplots()
    ax.plot(eps_evol)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance")
    ax.set_title("Evolution of distance to original point")
    plt.show()
    fig.savefig("eps_evol.png")
    return attack_info.values()

if __name__ == "__main__":
    
    
    
    X , y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Train model
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    # Predict
    y_pred = clf.predict(X_test)
    # Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("ROC AUC score:", roc_auc_score(y_test, y_pred))
    
    # Attack
    dist_func = lambda x, y: np.linalg.norm(x - y)
    print("SVM coef:", clf.coef_[0])
    # orig_point is random point from X_test
    orig_point = X_test[0]
    attack_info = fgsm_attack_svm_2c(clf, X_test[0], dist_function=dist_func)
    print("Attack info:", attack_info)
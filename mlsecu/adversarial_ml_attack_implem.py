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
def fgsm_attack_svm_2c(classifier:svm.SVC, orig_point, dist_function, step=None, epsilon=np.inf, max_step=200):
    data_point = orig_point.copy()
    orig_class = classifier.predict(data_point.reshape(1, -1))[0]
    new_class = orig_class
    current_eps = dist_function(data_point,orig_point)
    attack_info = None
    
    
    if step is None:
        step = 0.01
    i =0
    
    print("Original class:", orig_class)
    
    data_points_pos =[data_point]
    while orig_class == new_class:
        if current_eps < epsilon and i < max_step:
            grad =  classifier.coef_[0]
            data_point = data_point + step *  grad
            data_points_pos.append(data_point)
            new_class = classifier.predict(data_point.reshape(1, -1))[0]
            current_eps = dist_function(data_point,orig_point)
            attack_info = data_point, current_eps
        else:
            print("Attack failed")
            print("Current class:", new_class)
            print("Original class:", orig_class)
            attack_info = (None, None)
            
            if current_eps > epsilon:
                print("Attack failed: epsilon exceeded",current_eps)
            if i >= max_step:
                print("Attack failed: max step exceeded")
            print("Step:", i)
            break
            #return classifier.decision_function(data_point.reshape(1, -1))[0],0
        i += 1
    print("decision function",classifier.decision_function(data_point.reshape(1, -1))[0])
    fig, ax = plt.subplots(figsize=(10, 10))
    data_points_pos = np.array(data_points_pos)
    ax.plot(data_points_pos[:,0],data_points_pos[:,1], marker="x", linestyle="", label="Data points")
    ax.plot(orig_point[0],orig_point[1], marker="o", linestyle="", label="Original point")
    # plot the line which separates the two classes
    w = classifier.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1])
    yy = a * xx - (classifier.intercept_[0]) / w[1]
    ax.plot(xx, yy, 'k-')

    
    
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Decision boundary")
    ax.legend()
    fig.savefig("data_points_pos.png")
    return attack_info




if __name__ == "__main__":
    
    
    import random
    X , y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=10)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0,1000))
    # Train model
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    
    dist_func = lambda x, y: np.linalg.norm(x - y)
    
    orig_point = random.choice(X_test)
    
    attack_info = fgsm_attack_svm_2c(clf, orig_point, dist_function=dist_func,step=0.05)
    print("Attack info:", attack_info)
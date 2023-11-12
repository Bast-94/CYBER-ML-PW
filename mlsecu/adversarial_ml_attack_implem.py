import numpy as np
def fgsm_attack_svm_2c(classifier, orig_point, dist_function, step=None, epsilon=np.inf, max_step=200):
    data_point = orig_point.copy()
    orig_class = classifier.predict(data_point.reshape(1, -1))[0]
    new_class = orig_class
    current_eps = np.linalg.norm(data_point - orig_point)
    attack_info = {}
    if step is None:
        step = 0
    while orig_class == new_class:
        if current_eps < epsilon and step < max_step:
            # get datapoint by gradient descent
            grad = classifier.coef_[0]
            data_point = data_point + grad
            
            new_class = classifier.predict(data_point.reshape(1, -1))[0]
            current_eps = dist_function(data_point,orig_point)
            attack_info = dict(data_point=data_point, current_eps=current_eps)
        else:
            attack_info = dict(data_point=None, current_eps=None)
            if current_eps > epsilon:
                print("Attack failed: epsilon exceeded")
            if step >= max_step:
                print("Attack failed: max step exceeded")
            break
        step += 1
    return attack_info.values()

if __name__ == "__main__":
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
    attack_info = fgsm_attack_svm_2c(clf, X_test[0], dist_function=dist_func, epsilon=0.1)
    print("Attack info:", attack_info)
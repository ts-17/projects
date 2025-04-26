import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

url = "https://hastie.su.domains/ElemStatLearn/datasets/spam.data"
df = pd.read_csv(url, sep=" ", header=None)

X = df.iloc[:, :-1]  # features (all but last)
y = df.iloc[:, -1]   # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# first LDA to classify
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)

# then use QDA classification
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)

# then logistic regression
log_reg = LogisticRegression(solver="lbfgs", max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# and finally support vector machine
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# confusion matrix and error rates
def report_results(name, model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print(f"{name} - Train")
    cm_train = confusion_matrix(y_train, y_pred_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    print("Confusion Matrix:")
    print(cm_train)
    print(f"Accuracy: {acc_train:.4f}")
    print(f"Misclassification Rate: {1 - acc_train:.4f}")
    print("")

    print(f"{name} - Test")
    cm_test = confusion_matrix(y_test, model.predict(X_test))
    acc_test = accuracy_score(y_test, y_pred_test)
    print("Confusion Matrix:")
    print(cm_test)
    print(f"Accuracy: {acc_test:.4f}")
    print(f"Misclassification Rate: {1 - acc_test:.4f}")
    print("")

# prints
report_results("LDA", lda, X_train, y_train, X_test, y_test)
report_results("QDA", qda, X_train, y_train, X_test, y_test)
report_results("Logistic Regression", log_reg, X_train, y_train, X_test, y_test)
report_results("Support Vector Machines", svm, X_train, y_train, X_test, y_test)
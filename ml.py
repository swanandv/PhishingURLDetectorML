import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import confusion_matrix

def view_accuracies(file):

    dataset = pd.read_csv(file)

    #print(dataset.head())

    x = dataset.iloc[:,0:17].values
    y = dataset.iloc[:,17].values


    #oneHE=OneHotEncoder()
    #x=oneHE.fit_transform(x).toarray()
    #print("oneHE=")
    #print(x)


    #sc_X=StandardScaler()
    #x=sc_X.fit_transform(x)




    x_train, x_test , y_train, y_test = train_test_split(x,y,train_size=0.80)

    print("Training started")
    #from sklearn.linear_model import LogisticRegression
    #classifier = LogisticRegression()

    clf = [KNeighborsClassifier(),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        #GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        #QuadraticDiscriminantAnalysis(),
        #LogisticRegression(),
        #LinearRegression(),
        #DecisionTreeRegressor()
        ]

    names = [
        "Nearest Neighbors # ",
        "Linear SVM #",
        "RBF SVM #",
        #"Gaussian Process",
        "Decision Tree #",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        #"QDA",
        #"LogisticRegression #",
        #"Linear Regression",
        #"Decision Tree Regressor"
    ]
    
    for name,classifier in zip(names,clf):
        classifier.fit(x_train, y_train)
        print("Classifier name: "+name)
        y_pred = classifier.predict(x_test)
        score = classifier.score(x_test,y_test)
        print("Score: "+str(score))
        print()

        #if name == "Decision Tree #":
            #tree.export_graphviz(classifier,out_file="tree.dot")
            
        #Generating accuracy, precision, recall and f1-score
        print(confusion_matrix(y_test, y_pred))
        print()


#view_accuracies("A:\Research\Implementation\Dataset\extracted_features4.csv")









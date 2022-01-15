import functions
import ml
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

import os
here = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(here, 'Dataset\extracted_features4.csv')

dataset_file = filename #"Dataset\extracted_features4.csv"

while(True):
	ch =  int(input('''
Menu:
1. Scan URL
2. Train Model
3. View Accuracies of all models
4. Quit
Enter Choice:
	'''))
	if ch == 4:
		exit()
	
	if ch == 1:
		url = input("Enter URL to scan:")
		pred,row = functions.phishing_prediction(url)
		print(pred[0])
		if pred == [1]:
			print("Given URL is a Phishing URL")
		elif pred == [0]:
			print("Given URL is Safe")
		
		correct = input("Is the Prediction Correct?(Y/N/Skip):")
		if correct == "y" or correct  == "Y":
			row.append(pred[0])
			print(row)
			functions.new_entry(row,dataset_file)
			print("Line saved in Dataset")
		elif correct == "n" or correct == "N":
			if pred[0]==1:row.append("0")
			elif pred[0]==0:row.append("1")
			else: continue
			print(row)
			functions.new_entry(row,dataset_file)
			print("Corrected line saved in Dataset")

	if ch == 2:
		print(functions.train_model(SVC(gamma=2, C=1),dataset_file))

	if ch == 3:
		ml.view_accuracies(dataset_file)


	


		



from os import write
import pandas as pd
from urllib.parse import urlparse
from tldextract import extract
import re
import csv
import pandas as pd
import io
from shortening_services import *

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

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

import os
here = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(here, 'Dataset/')

def extract_features(link):
	url=link
	row=[]
    ## String length
    #headers.append("LenOfURL")
	no = len(url)
	if no > 1 and no <= 50:
		length=1
	elif no > 50 and no <= 100:
		length=2
	elif no > 100 and no <= 150:
		length=3
	elif no > 150 and no <= 200:
		length=4
	elif no > 200:
		length=5
	row.append(length)


	def url_parser(url):
		parts = urlparse(url)
		directories = parts.path.strip('/').split('/')
		queries = parts.query.strip('&').split('&')
		
		elements = {
			'scheme': parts.scheme,
			'netloc': parts.netloc,
			'path': parts.path,
			'params': parts.params,
			'query': parts.query,
			'fragment': parts.fragment,
			'directories': directories,
			'queries': queries,
		}
		
		return elements

	elements = url_parser(url)
	## http https:
	#headers.append("IsHTTPS")
	#print(elements.get('scheme'))
	if elements.get('scheme') == "https": row.append("1")
	elif elements.get('scheme') == "http": row.append("0")
	elif elements.get('scheme') == "": row.append("0")
	elif elements.get('scheme') == "ftp" : row.append("0")


	## Subdomain extract:
	domain = extract(url)
	#print(domain)
	#print(domain[0])

	## Path extract:
	path = str(elements.get('path'))
	#print("Path: "+path)

	## Directories extract:
	dir = str(elements.get('directories'))
	#print("Dir: "+str(dir))

	## Params extract:
	par = str(elements.get('params'))
	#print("Params: "+par)

	## Queries:
	q = str(elements.get("queries"))
	#print("Queries: "+str(q))

	##too many subdomains:
	#headers.append("ManySubdomains")
	#print(domain[0])
	subdomains = domain[0].split(".")
	#print(subdomains)
	row.append(len(subdomains))
	#if(len(subdomains)>2): 
		# print("too many subdomains") 
		#  row.append("1")
	#else: 
		#print("Less than 2 subdomains")
		#row.append("0")


	## Is IP?  ## regex to match ip
	#headers.append("IsIP")
	#print(domain[1])
	isip = 0
	#print(re.search("^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",domain[1]))
	if re.findall("^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",domain[1]): 
		#print("IP matched")
		row.append("1")
		isip = 1
	else: 
		#print("Domain Name")
		row.append("0")

	## no of parameters 
	#headers.append("ParamCount")  
	params = elements.get('queries')
	#print(len(params))
	row.append(len(params))

	## checking for shadow URL (another url in parameters)
	#headers.append("NestedURL")
	if re.search("[a-zA-z0-9]+\.(com|net|org|in|ie|uk|au|de)",dir) or re.search("[a-zA-z0-9]+\.(com|net|org|in|ie|uk|au|de)",par) or re.search("[a-zA-z0-9]+\.(com|net|org|in|ie|uk|au|de)",q): 
		#print("Nested URL found!")
		row.append("1")
	else:
		#print("Nested URL NOT found!")
		row.append("0")
	#print(re.search("[a-zA-z0-9]+\.(com|net|org|in|ie|uk|au|de)",url))

	## number of dashes
	#headers.append("DashCount")
	dashes = url.split("-")
	row.append(len(dashes)-1)


	## number of undrscore
	#headers.append("UnderscoreCount")
	undscore = url.split("_")
	row.append(len(undscore)-1)

	## to find email id in the URL:
	#headers.append("IsEmailId")
	if re.search("[a-zA-z0-9\.\-\_]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,3}",url) :
		#print("Email ID found in the URL")
		row.append("1")
	else:
		#print("Email ID NOT found in URL")
		row.append("0")


	## Uncommon top level domain:
	#headers.append("RareTLD")
	tlds = (filename+"most_common_tld.txt","r")
	flag = 0  
	for tld in tlds:
		if tld == domain[2]+"\n": 
			flag = 1     
			break        
		else: 
			flag = 0

	if flag == 1:
		#print("Common tld")
		row.append("0")
	elif isip == 1:
		#print("No TLD")
		row.append("0")
	elif flag == 0:
		#print("Rare tld")
		row.append("1")



	## Wordpress site?
	#headers.append("IsWordpress")
	wordpress = ["wp-admin","wp-content","wp-includes","theme-compat"]
	for i in wordpress:
		result = url.lower().find(i)
		if result > 0: break

	if result > 0: 
		#print("wordpress site")
		row.append("1")
	elif result == -1: 
		#print("Not a WP site")
		row.append("0")

	## famous brands in subdomain
	#headers.append("BrandsInSubdomain")
	file = open(filename+"famous_brands.txt","r")
	brands = file.readlines()
	flag1 = 0
	for brand in brands:
		if domain[0].lower().find(brand.rstrip()) > -1:
			flag1 = 1
			break
		
	if flag1 == 1: 
		#print("Famous brand found in subdomain")
		row.append("1")
	else:
		#print("Famous brand NOT found in subdomain")
		row.append("0")


	## famous brand in directories
	#headers.append("BrandsInDirectory")
	#print(elements.get("path"))


	flag2 = 0
	for brand in brands:
		if elements.get("path").lower().find(brand.rstrip()) > -1:
			#print("Famous brand matched in directory")
			flag2 = 1
			break
		
	if flag2 == 1:
		#print("Famous brand found in directory")
		row.append("1")
	else:
		#print("Fomous brand NOT found in directory")
		row.append("0")



	### Free-form free-hosting sites
	#headers.append("FreeHosting")
	hostingfile =  open(filename+"free_hosting_domains.txt","r")
	sites = hostingfile.readlines()
	flag3=0
	for site in sites:
		if url.lower().find(site.rstrip()) > -1:
			flag3=1
			break

	if flag3 == 1:
		#print("Free-hosting site found")
		row.append("1")
	else:
		#print("Free-hosting site NOT found")
		row.append("0")



	## Famous brand in domain
	file = open(filename+"famous_brands.txt","r")
	brands = file.readlines()
	flag4 = 0
	for brand in brands:
		if domain[1].lower().find(brand.rstrip(),0,len(domain[1])) > -1:
			flag4 = 1
			break
		else:
			flag4 = 0

	if flag4 == 1: 
		#print("Famous Brand in Domain")
		row.append("1")
	else: 
		row.append("0")


	## Shortner Service
	if re.search(shortening_services, url):
		row.append("1")
		#print("Shortening service Found")
	else:
		row.append("0")



	## Slash count
	slashes  = re.findall("/",url)
	row.append(len(slashes))




	return row




def save_model(model):
	filenames = 'saved_ml_model.sav'
	pickle.dump(model, open(filenames, 'wb'))
	print("Model saved to file.")

	##pickle.dump(x_scaled,open('saved_scaler.sav','wb'))
	#print('Scaler saved to file.')


def train_model(classifier,file):
	dataset = pd.read_csv(file)
	x = dataset.iloc[:,0:17].values
	y = dataset.iloc[:,17].values


	#oneHE=OneHotEncoder()
	#x=oneHE.fit_transform(x).toarray()

	#sc_X=StandardScaler()
	#x_sc=sc_X.fit_transform(x)
	#print("x after SC:")
	#print(x_sc)
	x_train, x_test , y_train, y_test = train_test_split(x,y,train_size=0.80)
	print("Training started")
	classifier.fit(x_train, y_train)

	#importance = classifier.feature_importances_
# summarize feature importance
	#for i,v in enumerate(importance):
		#print('Feature: %0d, Score: %.5f' % (i,v))


	score = classifier.score(x_test,y_test)
	y_pred = classifier.predict(x_test)
	print(classification_report(y_test,y_pred))
	print("Score: "+str(score))

	#training using full dataset before saving the model
	x_train, x_test , y_train, y_test = train_test_split(x,y,train_size=0.99)
	classifier.fit(x_train, y_train)

	save_model(classifier)
	return "Model Training"


def load_model():
	filenames = 'saved_ml_model.sav'
	loaded_model = pickle.load(open(filenames, 'rb'))
	print("Model Loading Successful")
	return loaded_model
	#loaded_scaler =  pickle.load(open('saved_scaler.sav','rb'))
	#print("Scaler Loading Successful")
	#return loaded_model,loaded_scaler




#model=train_model(AdaBoostClassifier())
#load_model()

def phishing_prediction(url):
	newurl = extract_features(url)
	classifier = load_model()

	print(classifier)
	
	x_new =  np.array(newurl).reshape(1,-1)



	print(x_new)
	y_new = classifier.predict(x_new)
	
	print("Prediction:")
	print(y_new)

	return y_new,newurl

def new_entry(row,file):
	outfile = io.open(file,'a',encoding="utf-8",newline="")
	writer = csv.writer(outfile)
	writer.writerow(row)
	

	
	







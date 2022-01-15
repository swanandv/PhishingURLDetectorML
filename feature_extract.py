from os import write
import pandas as pd
from urllib.parse import urlparse
from tldextract import extract
import re
import csv
import pandas as pd
import io
from  shortening_services import shortening_services

headers = ['LenOfURL', 'IsHTTPS', 'SubdomainsCount', 'IsIP', 'ParamCount', 'NestedURL', 'DashCount', 'UnderscoreCount', 'IsEmailId', 'RareTLD', 'IsWordpress', 'BrandsInSubdomain', 'BrandsInDirectory', 'FreeHosting','BrandInDomain','ShorteningService','SlashCount', 'Phishing']

outfile = io.open('A:\Research\Implementation\Dataset\extracted_features4.csv','w',encoding="utf-8",newline="")
writer = csv.writer(outfile)

writer.writerow(headers)

csv_file = io.open('A:\Research\Implementation\Dataset\dataset_shuffled.csv',encoding="utf-8")
csv_reader = csv.reader(csv_file, delimiter=',')
count=0

for line in csv_reader:
    url = line[0].lstrip()
    count += 1
    print("----------------------------------------------------------------------------------------")
    print(url)

#data = pd.read_csv("A:\Research\Implementation\Dataset\\dataset_test_20rows.csv",encoding="utf-8",)
#print(data)


#url = "http://abc.paypal.xyz.flyandlure.xyz/wp-admin/facebook.comarticles/aib/fly_fishing1abc@abc.in/fbly_fishing_diary_july_2020?q=word&b=something&c=asd&d=dsacndnd&e=dakmd123#anchor"
#url = "https://10.20.30.123:80/index.php?a=pqr"
#url = "123formbuilder.com/sdasdasdsdasd/sdasdasdsdd"


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
    print(elements.get('scheme'))
    if elements.get('scheme') == "https": row.append("1")
    elif elements.get('scheme') == "http": row.append("0")
    elif elements.get('scheme') == "": row.append("0")
    elif elements.get('scheme') == "ftp" : row.append("0")


    ## Subdomain extract:
    domain = extract(url)
    print(domain)
    print(domain[0])

    ## Path extract:
    path = str(elements.get('path'))
    print("Path: "+path)

    ## Directories extract:
    dir = str(elements.get('directories'))
    print("Dir: "+str(dir))

    ## Params extract:
    par = str(elements.get('params'))
    print("Params: "+par)

    ## Queries:
    q = str(elements.get("queries"))
    print("Queries: "+str(q))

    ##too many subdomains:
    #headers.append("ManySubdomains")
    print(domain[0])
    subdomains = domain[0].split(".")
    print(subdomains)
    row.append(len(subdomains))
    #if(len(subdomains)>2): 
       # print("too many subdomains") 
      #  row.append("1")
    #else: 
        #print("Less than 2 subdomains")
        #row.append("0")


    ## Is IP?  ## regex to match ip
    #headers.append("IsIP")
    print(domain[1])
    isip = 0
    print(re.search("^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",domain[1]))
    if re.findall("^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",domain[1]): 
        print("IP matched")
        row.append("1")
        isip = 1
    else: 
        print("Domain Name")
        row.append("0")

    ## no of parameters 
    #headers.append("ParamCount")  
    params = elements.get('queries')
    print(len(params))
    row.append(len(params))

    ## checking for shadow URL (another url in parameters)
    #headers.append("NestedURL")
    if re.search("[a-zA-z0-9]+\.(com|net|org|in|ie|uk|au|de)",dir) or re.search("[a-zA-z0-9]+\.(com|net|org|in|ie|uk|au|de)",par) or re.search("[a-zA-z0-9]+\.(com|net|org|in|ie|uk|au|de)",q): 
        print("Nested URL found!")
        row.append("1")
    else:
        print("Nested URL NOT found!")
        row.append("0")
    print(re.search("[a-zA-z0-9]+\.(com|net|org|in|ie|uk|au|de)",url))

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
        print("Email ID found in the URL")
        row.append("1")
    else:
        print("Email ID NOT found in URL")
        row.append("0")


    ## Uncommon top level domain:
    #headers.append("RareTLD")
    tlds = open("A:\Research\Implementation\Dataset\most_common_tld.txt","r")
    flag = 0  
    for tld in tlds:
        if tld == domain[2]+"\n": 
            flag = 1     
            break        
        else: 
            flag = 0

    if flag == 1:
        print("Common tld")
        row.append("0")
    elif isip == 1:
        print("No TLD")
        row.append("0")
    elif flag == 0:
        print("Rare tld")
        row.append("1")



    ## Wordpress site?
    #headers.append("IsWordpress")
    wordpress = ["wp-admin","wp-content","wp-includes","theme-compat"]
    for i in wordpress:
        result = url.lower().find(i)
        if result > 0: break

    if result > 0: 
        print("wordpress site")
        row.append("1")
    elif result == -1: 
        print("Not a WP site")
        row.append("0")

    ## famous brands in subdomain
    #headers.append("BrandsInSubdomain")
    file = open("A:\Research\Implementation\Dataset\\famous_brands.txt","r")
    brands = file.readlines()
    flag1 = 0
    for brand in brands:
        if domain[0].lower().find(brand.rstrip()) > -1:
            flag1 = 1
            break
        
    if flag1 == 1: 
        print("Famous brand found in subdomain")
        row.append("1")
    else:
        print("Famous brand NOT found in subdomain")
        row.append("0")


    ## famous brand in directories
    #headers.append("BrandsInDirectory")
    print(elements.get("path"))


    flag2 = 0
    for brand in brands:
        if elements.get("path").lower().find(brand.rstrip()) > -1:
            print("Famous brand matched in directory")
            flag2 = 1
            break
        
    if flag2 == 1:
        print("Famous brand found in directory")
        row.append("1")
    else:
        print("Fomous brand NOT found in directory")
        row.append("0")

    

    ### Free-form free-hosting sites
    #headers.append("FreeHosting")
    hostingfile =  open("A:\Research\Implementation\Dataset\\free_hosting_domains.txt","r")
    sites = hostingfile.readlines()
    flag3=0
    for site in sites:
        if url.lower().find(site.rstrip()) > -1:
            flag3=1
            break

    if flag3 == 1:
        print("Free-hosting site found")
        row.append("1")
    else:
        print("Free-hosting site NOT found")
        row.append("0")



    ## Famous brand in domain
    file = open("A:\Research\Implementation\Dataset\\famous_brands.txt","r")
    brands = file.readlines()
    flag4 = 0
    for brand in brands:
        if re.search("^"+brand.strip()+"$",domain[1]):
            flag4 = 1
            break
        else:
            flag4 = 0
    
    
    if flag4 == 1: 
        print("Famous Brand in Domain")
        row.append("1")
    else: 
        row.append("0")


    ## Shortner Service
    if re.search(shortening_services, url):
        row.append("1")
        print("Shortening service Found")
    else:
        row.append("0")
    


    ## Slash count
    slashes  = re.findall("/",url)
    row.append(len(slashes))



    ## Adding target variable 
    #headers.append("Phishing")
    if line[1] == "Phish":
        row.append("1")
    elif line[1] == "Benign":
        row.append("0")


    print(headers)
    print(len(headers))
    print(row)
    print(len(row))

    row[14] = row[14].rstrip()

    writer.writerow(row)
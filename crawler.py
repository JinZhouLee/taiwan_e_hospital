"""
This file is used to crawl data from 台灣e院 (sp1.hso.mohw.gov.tw). 
Runs in Python 3.12.4.

@author: Jin-Zhou Lee
"""

#%% pacakges
import requests # for web crawler, send request to web
from bs4 import BeautifulSoup # html read tool
from urllib.parse import urlparse, parse_qs
import warnings 
import time
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import os

#%% crawler setting
# use ipv4 to boost
requests.packages.urllib3.util.connection.HAS_IPV6 = False
# disable SSH warning
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
wait_time = 0.5 # waiting time to access each web page
timeout = 5 # timeout for each request (in seconds)
max_retries = 5 # maximum number of retries

#%% url to html function
def url_to_html(url):
    for retrie in range(max_retries): 
        try:
            html = requests.get(url, verify = False, timeout = timeout) 
            # send an access request to the specified web page
            html.encoding = 'big5' # read the website in big5 format
            return html.text
        except:
            time.sleep(wait_time)
    return " "

def html_to_soup(html):
    soup = BeautifulSoup(html, 'html.parser') # analysis html
    return soup

def url_to_soup(url):
    html = url_to_html(url)
    soup = BeautifulSoup(html, 'html.parser') # analysis html
    return soup

#%% EDA
# 
# 
# 

#%% crawler often question
def fetch_often_question():
    global often_question
    
    # often question url
    url = "https://sp1.hso.mohw.gov.tw/doctor/Often_question/index.php"
    # crawle it
    soup = url_to_soup(url) 
    # result 
    often_question = pd.DataFrame()

    # find the departments 
    text = soup.find_all(class_ = 'w3-padding w3-hide-large w3-hide-medium')
    text = [t.get_text() for t in text]
    often_question["科別"] = text

    # find the department often questions
    text = soup.find_all(class_ = 'co-td')
    text = [[quest.get_text() for quest in t.find_all("a")] for t in text]
    often_question["問題"] = text
    often_question["問題種類"] = [len(t) for t in text]
    
fetch_often_question()

#%% crawler questions number in each department
def fetch_questions():
    url_base = "https://sp1.hso.mohw.gov.tw/doctor/All/history.php?UrlClass="

    for i in range(len(often_question)):
        # the url of each department 
        url = url_base + often_question.loc[i, "科別"]
        soup = url_to_soup(url)
        quest_num = soup.find(style = "color:red;").get_text()
        often_question.loc[i, "問題數量"] = int(quest_num)
        
        # progress Bar
        print(i + 1, "/", len(often_question), "\t", (i + 1) / len(often_question) * 100, "%")
        
        # sleep
        time.sleep(wait_time)

    often_question.to_pickle("data/often_question.pkl")
    
fetch_questions()

#%%

#%% EDA often question
#
#
#

#%% 問題種類 EDA
often_question = pd.read_pickle("data/often_question.pkl")

def plt_question_type():
    plt.figure(figsize=(19.2, 10.8)) # set figure size

    plt.rc('font', family='Microsoft JhengHei') # set chinese font
    plt.xticks(fontsize=16)  # set x-axe font size
    plt.yticks(fontsize=16)  # set y-axe font size
    
    plt.title("各科問題種類", fontsize=24) # figure title
    
    # sorting data
    often_question_sorted = often_question.sort_values(by='問題種類', ascending = False)
    plt.barh(often_question_sorted['科別'], often_question_sorted['問題種類']) # figure data
    plt.tight_layout()  # figure layout
    
    # save figure
    plt.savefig("img/各科問題種類.png", dpi=200, bbox_inches='tight') 
    
plt_question_type()

#%% 問題數量 EDA
def plt_question_num():
    plt.figure(figsize=(19.2, 10.8)) # set figure size

    plt.rc('font', family='Microsoft JhengHei') # set chinese font
    plt.xticks(fontsize=16)  # set x-axe font size
    plt.yticks(fontsize=16)  # set y-axe font size
    
    plt.title("各科問題數量", fontsize=24) # figure title
    
    # sorting data
    often_question_sorted = often_question.sort_values(by='問題數量', ascending = False)
    plt.barh(often_question_sorted['科別'], often_question_sorted['問題數量']) # figure data
    plt.tight_layout()  # figure layout
    
    # save figure
    plt.savefig("img/各科問題數量.png", dpi=200, bbox_inches='tight') 

plt_question_num()

#%% crawler content
#
#
#

#%% crawl single department
def fetch_depart_pages(depart, max_page = 10):
    '''
    depart: select department
    max_page: The maximum number of pages to read (20 records per page)
    '''
    
    # result data frame
    data = pd.DataFrame()

    # in each page
    for page in range(1, max_page + 1):
        # page url
        url = "https://sp1.hso.mohw.gov.tw/doctor/All/history.php?UrlClass=" + depart + "&SortBy=q_no&PageNo=" + str(page)
        # crawler page
        html = url_to_html(url)
        soup = BeautifulSoup(html, 'html.parser')
        
        # read table
        table = pd.read_html(StringIO(html))[0]
        
        # get url
        links = soup.find("table").find_all('a')
        link = [link.get('href') for link in links]
        q_no = [parse_qs(urlparse(link[i]).query).get('q_no', [None])[0] for i in range(len(table))]
        
        # get depart
        table["q_no"] = q_no
        table["科別"] = depart
        
        # merge
        data = pd.concat([data, table])
        
        # progress bar
        print(depart, page, "/", max_page, "\t", page / max_page * 100, "%")
    
    return data

#%% crawl each department
# max number of department 
max_depart = 10
# max number of pages to read
max_page = 50

def fetch_depart():
    # select the dapartment to read
    often_question = pd.read_csv("data/often_question.csv")
    often_question = often_question.sort_values(by='問題數量', ascending = False)
    select_dapart = often_question['科別'][:max_depart]
    
    # crawling data and stroge to data/depart
    for part in select_dapart:
        data = fetch_depart_pages(part, max_page = max_page)
        data.to_csv("data/depart/" + part + '.csv', index = False)

fetch_depart()
    
#%% merge csv file
def marge_fetch_depart():
    global questions
    
    # read data from data/depart
    folder_path = 'data/depart/'
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # stack multiple csv file
    dataframes = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]
    # stack into one dataframe
    questions = pd.concat(dataframes, ignore_index = True)
    # sorting by q_no
    questions = questions.sort_values(by='q_no', ascending = True)

#%% crawl question content
def fetch_content():
    for i in range(len(questions)):
        # question url
        url = "https://sp1.hso.mohw.gov.tw/doctor/All/ShowDetail.php?q_no=" + str(questions["q_no"].iloc[i])
        
        # crawl it
        soup = url_to_soup(url)
        # get content
        content = soup.find(class_ = "msg").find("div").text
        # put it to the dataframe
        questions.loc[i, "內文"] = content
        
        # progress bar
        print(i + 1, "/", len(questions), "\t", (i + 1) / len(questions) * 100, "%")
        
        # waiting
        time.sleep(wait_time)
        
        # save data
        questions.to_csv("data/content.csv", index = False)

marge_fetch_depart() 
fetch_content()

#%% clear by-products data
import shutil
shutil.rmtree('data/depart')

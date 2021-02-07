# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import time
import sys
from selenium import webdriver
from bs4 import BeautifulSoup
import re
from collections import defaultdict
import requests
import pandas as pd
import numpy as np
import pandas as pdC
import pickle
import json
from urllib.request import urlopen
with open("labeled_names.pkl", "rb") as f:
    labeled_names = pickle.load(f)

labeled_names = [(x[0].lower(), x[1]) for x in labeled_names]
labeled_names = dict(labeled_names)
print(len(labeled_names))

from keras.models import load_model

PATH = 'C:/Users/Laurin/Downloads/chromedriver_win32/chromedriver.exe'  # Optional argument, if not specified will search path.
MODEL_PATH = './model'

chars = {
    ' ', "'", '-', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
}
char_indices = dict((c, i) for i, c in enumerate(chars))

model = load_model(MODEL_PATH)

maxlen = model.layers[0].input_shape[1]


def get_feature_vec(names: list):
    if len(names) > maxlen:
        names = names[:15]
    X = np.zeros((len(names), maxlen, len(chars)), dtype=np.bool)
    for i, name in enumerate(names):
        for t, char in enumerate(name.lower()):
            try:
                X[i, t, char_indices[char]] = 1
            except KeyError:
                continue
    return X   


def scroll(driver, timeout):
    scroll_pause_time = timeout

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(scroll_pause_time)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            # If heights are the same it will exit the function
            break
        last_height = new_height


# +
def render_page(url):
    driver = webdriver.Chrome(PATH)
    driver.get(url)
    time.sleep(3)
    scroll(driver, 4)
    r = driver.page_source
    return r

anker = "https://www.ycombinator.com/companies"

r = render_page(anker)

soup = BeautifulSoup(r, "html.parser")


# -

def get_frac_male_founders(doc):
    founders = doc.findAll("div", {"class": "founder-info flex-row"})
    if len(founders) == 0:
        return np.nan, []
    names = [founder.find("h3", {"class": "heavy"}).contents[0].split(" ")[0] for founder in founders]
    males = []
    for name in names:
        if name.lower() in list(labeled_names.keys()):
            males.append(int(labeled_names[name.lower()]=="male"))
        else:
            continue
            males.append(1-np.argmax(model.predict(get_feature_vec([name]))))  # male is indicated with 0 in the model
    if len(males) > 0:
        return sum(males) / len(males), names
    else:
        return np.nan, []


tech_words = [
    "digital", "marketplace", "tech", "technology", "platform", "onlin", "mobile", "internet", "ai", 
    "cloud", "artificial intelligence", "computer", "app", "network", "server", "automat", "automiz",
    "analytics", "software", "e-commerce", "saas", "developer"
]

database = defaultdict(list)
comp_name = list()
for i, company in enumerate(soup.findAll('a', href=re.compile('/companies/\d'))):
    # determine stuff
    comp_id = company.attrs["href"].split("/")[-1]
    url = anker + "/" + comp_id if not anker[-1] == "/" else anker + comp_id
    
    # scrape company site
    r = requests.get(url)
    doc = BeautifulSoup(r.text, "html.parser")
    
    # scan if is tech comp
    try:
        category = ", ".join([cat.contents[0] for cat in doc.findAll("span", {"class": "pill"}) if isinstance(cat.contents[0], str)])
        headline = doc.find("h3").contents[0]
        tech = any([word in " ".join([
            doc.find("p", {"class": "pre-line"}).contents[0], headline
        ]) for word in tech_words])
    except IndexError:
        tech = np.nan
    
    frac_male, names = get_frac_male_founders(doc)
    # add to database
    database["comp_id"].append(comp_id)
    database["comp_name"].append(company.find("span", {"class": "SharedDirectory-module__coName___gbFfW"}).contents[0]) 
    database["url"].append(url)
    database["is_tech"].append(tech)
    database["frac_male"].append(frac_male)
    database["names"].append(", ".join(names))
    
    if (i+1) % 1000 == 0:
        break

df = pd.DataFrame.from_dict(database)

print(df.is_tech.mean(skipna=True))

print(df.frac_male.mean(skipna=True))

df["num_founders"] = df.names.apply(lambda x: len(x.split(", ")))
df["weight"] = df["num_founders"] / df["num_founders"].sum()
print((df["weight"] * df.frac_male).sum())

df[df.frac_male < 1]

df.to_csv("database.csv", sep=";", decimal=",")

# ## Get gender from API

names = ", ".join(df.names.to_list())
names = names.split(", ")
names = [name.lower() for name in names]
print(len(names))

new_names = []
for name in names:
    if any([not c.isalnum() for c in name]):
        print(name)
    else:
        try:
            name.encode("ascii")
            new_names.append(name)
        except:
            print(name)
            continue

try:
    new_labeled_names
except NameError:
    new_labeled_names = []
else:
    new_names = list(set(new_names).difference([name[0] for name in new_labeled_names]))

len(new_names)

for i, name in enumerate(new_names):
    if i > 500:
        break
    name = name.lower()
    myKey = "zDzVBdLcGmCqjoLPqm"
    url = "https://gender-api.com/get?key=" + myKey + f"&name={name}"
    response = urlopen(url)
    decoded = response.read().decode('utf-8')
    data = json.loads(decoded)
    new_labeled_names.append((name, data["gender"]))

aaa = new_labeled_names[:]

sum([aa[1] == "male" for aa in aaa]) / len(aaa)

labeled_names.extend(new_labeled_names)

with open("labeled_names.pkl", "wb") as f:
    pickle.dump(new_labeled_names, f, pickle.HIGHEST_PROTOCOL)

with open("labeled_names.pkl", "rb") as f:
    huhu = pickle.load(f)

len(huhu)

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import requests
from bs4 import BeautifulSoup


def find_synonym(des):
    des = des.split(' ')
    des = '-'.join(des)
    link = f'https://abadis.ir/fatofa/{des}/'
    response = requests.get(link)
    soup = BeautifulSoup(response.text)
    synonyms = soup.find('div', {'t': 'مترادف ها'})
    synonyms_list = synonyms.find_all("div", {"class": None})
    possibleAnswers = [elem.text for elem in synonyms_list]
    print(possibleAnswers)


def classify(string):
    df = pd.read_csv('classification_data.csv')
    df = df.sample(frac=1)
    temp_df = pd.DataFrame({"شرح": [string], "دسته": ['30']})
    df1 = df.append(temp_df, ignore_index=True)
    x = df1['شرح']
    y = df1['دسته']
    for st in x:
        st.replace('از', '')
        st.replace('به', '')
        st.replace('در', '')

    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(x)
    size = x.shape[0]
    X_train, X_test = np.split(x.toarray(), [size - 1])
    y_train = y[0:size - 1].astype('int')

    model = MultinomialNB()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return prediction[0]


def possible_answers(Class, length):
    df = pd.read_csv('G:/uni/term5/AI/pro/answers/' + str(Class) + '.csv')
    ans = []
    df = df['پاسخها'].tolist()
    for x in df:
        y = x.replace(' ', '')
        if len(y) == int(length):
            ans.append(x)
    return ans

# stringsyn = input('enter the worl you want to find syn of: ')
# print(find_synonym(stringsyn))
string = input('enter your string: ')
Class = classify(string)
print('class: ' + str(Class))
length = input('enter length: ')
ans = possible_answers(Class, length)
for i in ans:
    print(i)

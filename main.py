from prettytable import PrettyTable
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


def promising_fill(table, row, column, op, length, answer):
    if op == 'vertical':
        for i in range(row, row + length):
            if table.rows[i][column] == '#' or table.rows[i][column] != answer[i - row]:
                return False
            else:
                table.rows[i][column] = answer[i - row]
    else:
        for j in range(column, column + length):
            if table.rows[row][j] == '#' or table.rows[row][j] == answer[j - row]:
                return False
            else:
                table.rows[row][j] = answer[j - row]
    return True


def backtrack_csp(table, row, column, op, length, questions, current_question):
    Class = classify(questions[current_question])
    ans = possible_answers(Class, length[current_question])
    for an_answer in ans:
        if promising_fill(table, row, column, op, length[current_question], an_answer):
            if current_question == len(questions) - 1:
                print(table)
                return
            else:
                if op == 'vertical':
                    backtrack_csp(table, row + length, column, op, length[current_question + 1], questions,
                                  current_question + 1)
                else:
                    backtrack_csp(table, row, column + length, op, length[current_question + 1], questions,
                                  current_question + 1)


print('Enter the size of puzzle: ')
row = int(input())
column = int(input())
print('So how you want to generate your puzzle: ')
binaryString = input()
table_shape = np.empty((row, column), dtype=object)
for i in range(row):
    for j in range(column):
        if binaryString[j + i * column] == "0":
            table_shape[i][j] = "0"
        else:
            table_shape[i][j] = "#"

table = PrettyTable()
for i in range(row):
    table.add_row(table_shape[i])
print(table)

questions = input('Enter the questions: ')
questions = questions.replace('&', '')
row_questions_list = questions.split('@')
all_questions_list = []
for x in row_questions_list:
    all_questions_list.append(x.split('#'))
length = int(len(binaryString.split('1')[0]))
backtrack_csp(table, 0, 0, 'horizontal', length, all_questions_list, 0)

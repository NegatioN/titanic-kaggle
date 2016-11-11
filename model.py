import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def reMapAsInt(dataframe, indice, value, newValue):
    print("value={} reassigned to id={}".format(value, newValue))
    dataframe.loc[dataframe[indice] == value, indice] = int(newValue)

def fillNaN(dataframe, indice, value):
    print("reassigning all NaN in {} to {}".format(indice, value))
    dataframe[indice] = dataframe[indice].fillna(value)

def massage_data(dataframe):
    fillNaN(dataframe, 'Embarked', train_df['Embarked'].value_counts().idxmax()) # set missing port to most common
    fillNaN(dataframe, 'Age', train_df['Age'].median()) # set missing age to median
    # remap sex and ports to numericals
    for id, port in embarked_list:
        reMapAsInt(dataframe, 'Embarked', port, id)
    for id, gender in sex_list:
        reMapAsInt(dataframe, 'Sex', gender, id)

def accuracy(predictions, answers):
    return sum(answers == predictions)/len(predictions)

train_df = pd.read_csv('titanic-data/train.csv', header=0)
test_df = pd.read_csv('titanic-data/test.csv', header=0)

fillNaN(train_df, 'Embarked', train_df['Embarked'].value_counts().idxmax()) # set missing port to most common
fillNaN(train_df, 'Age', train_df['Age'].median()) # set missing age to median
embarked_list = list(enumerate(np.unique(train_df['Embarked'])))
sex_list = list(enumerate(np.unique(train_df['Sex'])))

massage_data(train_df)
massage_data(test_df)


evaluated_indices = ['Sex', 'Age', 'Embarked']

algo = LinearRegression()
algo.fit(train_df[evaluated_indices], train_df['Survived'])

predictions = algo.predict(train_df[evaluated_indices])
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

print(algo.predict(train_df[evaluated_indices]))
print(accuracy(predictions, train_df['Survived']))






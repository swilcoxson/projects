import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


data = pd.read_csv("/Users/samanthawilcoxson/Documents/Projects/pokemon/pokemon.csv")

data = data.loc[data['is_legendary'] == 0]
maj_male = []
for i in range(len(data)):
    value = data['percentage_male'].iloc[i]
    if value > 50:
        maj_male.append(1)
    else:
        maj_male.append(0)

data['maj_male'] = pd.Series(maj_male)

features = ['name', 'type1', 'type2', 'maj_male']

pokemon = data[features]    #orig features are stored in pokemon
data = pd.get_dummies(pokemon, columns = ['type1', 'type2'])


train, val = train_test_split(data.dropna(), test_size = 0.2)

features = list(data.columns)
features.remove('name')
features.remove('maj_male')
target = 'maj_male'

lab_enc = preprocessing.LabelEncoder()
train[target] = lab_enc.fit_transform(train[target])
val[target] = lab_enc.fit_transform(val[target])

print(train[:5])
print(val[:5])


knn = KNeighborsClassifier(n_neighbors = 2)

knn.fit(train[features], train[target])

print(knn)

pred = knn.predict(val[features])

from sklearn.metrics import accuracy_score

train_results = train
train_results.drop('maj_male', axis=1)
train_results['pred'] = knn.predict(train[features])
train_results['actual'] = train[target]

score = accuracy_score(train[target], train_results['pred'])

print(train_results[:15])
print(score)

'''

score = accuracy_score(val[target], pred)

print(score)

val_results = val.drop('maj_male', axis=1)

val_results['pred'] = pd.Series(pred)
val_results['actual'] = val[target]

print(val_results[:15])

'''
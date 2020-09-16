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

score = accuracy_score(val[target], pred)

print(score)

'''

origins = pd.read_csv("/Users/samanthawilcoxson/Documents/Projects/pokemon/Pokemon-Origins-master/data.csv")
features = ['name', 'origin', 'generation', 'is_legendary', 'percentage_male', 'type1']
#target = ['type1']

features_trimmed = ['name', 'origin', 'type1']
features = features_trimmed

column = []


for i in range(len(data)):
    column.append(origins.iloc[i]['Origin'])

to_one_hot_encode = ['origin']

data['origin'] = pd.Series(column)

pokemon = pd.get_dummies(data[features], columns = to_one_hot_encode)


train, val = train_test_split(pokemon, test_size=0.2) 
target_train = train['type1']
target_val = val['type1']

print(train[:5])
print(val[:5])

train = train.drop(['name', 'type1'], axis = 1)
val = val.drop(['name', 'type1'], axis = 1)

'''

'''
research questions:
    - correlation between type and gender?
    - generation and type?
'''


'''
knn = KNeighborsClassifier(n_neighbors=16)
impute = KNNImputer(n_neighbors=16)
train_filled = impute.fit_transform(train)
val_filled = impute.fit_transform(val)
knn.fit(train_filled, target_train)
pred = knn.predict(val_filled)



score = accuracy_score(target_val, pred)

print("accuracy: "+str(score))

'''
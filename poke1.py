import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier


data = pd.read_csv("/Users/samanthawilcoxson/Documents/Projects/pokemon/pokemon.csv")
data = data.loc[data['generation'] == 1]  #only first gen
maj_male = []
data = data.fillna(value=-1.0)
for i in range(len(data)):
    value = data.iloc[i]['percentage_male']
    if value >= 60.0:
        maj_male.append('m')
    elif value <= 40.0:
        maj_male.append('f')
    elif value == -1.0:
        maj_male.append('n/a')        
    else:
        maj_male.append('n')

data['maj_male'] = pd.Series(maj_male)


features = ['name', 'type1', 'type2', 'maj_male', 'percentage_male']

pokemon = data[features]    #orig features are stored in pokemon
data = pd.get_dummies(pokemon, columns = ['type1', 'type2'])
data['type1'] = pokemon['type1']
data['type2'] = pokemon['type2']
data['percentage_male'] = pokemon['percentage_male']

#train, val = train_test_split(data.dropna(), test_size = 0.2)
train, val = train_test_split(data, test_size = 0.2, random_state=38)


features = list(data.columns)
features.remove('name')
features.remove('maj_male')
features.remove('type1')
features.remove('type2')
features.remove('percentage_male')
target = 'maj_male'

'''
lab_enc = preprocessing.LabelEncoder()
train[target] = lab_enc.fit_transform(train[target])
val[target] = lab_enc.fit_transform(val[target])
'''

'''
print(train[:5])
print(val[:5])
'''

knn = KNeighborsClassifier(n_neighbors = 2)

knn.fit(train[features], train[target])

from sklearn.metrics import accuracy_score

train_results = train
train_results = train_results.drop('maj_male', axis=1)
train_results['pred'] = knn.predict(train[features])
train_results['actual'] = train[target]

score = accuracy_score(train[target], train_results['pred'])

print(train_results[:15])
print(score)



val_results = val.drop('maj_male', axis=1)
val_results = val_results.drop(features, axis=1)
val_results['pred'] = knn.predict(val[features])
val_results['actual'] = val[target]
val_results['percentage_male'] = val['percentage_male']

score = accuracy_score(val[target], val_results['pred'])


print(val_results[:15])
print(score)


dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(train[features], train[target])
print("baseline accuracy: "+str(accuracy_score(val[target], dummy.predict(val[features]))))
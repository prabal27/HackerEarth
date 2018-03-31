# Classification template

# Importing the libraries
import numpy as n
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('criminal_train.csv')
X_train = dataset.iloc[:,1:71].values
y_train = dataset.iloc[:,71].values
X_test_dataset=pd.read_csv('criminal_test.csv')
X_test=X_test_dataset.iloc[:,1:71].values

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN 
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 128 , kernel_initializer = 'uniform', activation = 'relu', input_dim = 70))
classifier.add(Dropout(p = 0.3))

# Adding the second hidden layer
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.3))

## Adding the second hidden layer
#classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dropout(p = 0.3))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


y_pred_categorical=[]
for value in y_pred:
    if (value==True):
        y_pred_categorical.append('1')
    else:
         y_pred_categorical.append('0')

output=pd.read_csv('criminal_test.csv',usecols=[0])
output['Criminal']=y_pred_categorical
output.to_csv('output.csv', encoding='utf-8', index=False)

         
          
         
         
         


output=[]
for value in y_pred:
    print (y_pred)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 80, kernel_initializer = 'uniform', activation = 'relu', input_dim = 70))
    classifier.add(Dropout(p = 0.3))
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.3))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [10,25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
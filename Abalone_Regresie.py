from sklearn import neural_network
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt

# INCARCARE DATE
data = pd.read_csv('abalone.data')
data['sex'] = data['sex'].astype('category') 

categ = np.array(data['sex'].cat.codes.values).reshape(-1,1) 
conts = np.array(data.drop(['sex','rings'],axis=1)) 

y = np.array(data['rings'].values) 

size = len(data['sex'])
train_size = int(0.75*size)
test_size=int(0.25*size)

#IMPARTIRE IN TRAIN SI TEST
categ_train = categ[:train_size]
categ_test = categ[train_size:]

cont_train = conts[:train_size]
cont_test = conts[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print('Date de antrenare:\n', np.append(categ_train,cont_train,axis=1))
print('Etichete de antrenare:\n', y_train)
print('\nDate de testare:\n', np.append(categ_test,cont_test,axis=1))
print('Etichete de testare:\n', y_test)

# CREARE SI ANTRENARE MLP
regr = neural_network.MLPRegressor(hidden_layer_sizes=(8,4),max_iter=2000,verbose=False,learning_rate_init=0.01) #verbose=True pentru a vedea cum se antreneaza
regr.fit(np.append(categ_train,cont_train,axis=1),y_train)

# TESTARE MLP
predictii = regr.predict(np.append(categ_test,cont_test,axis=1))

# predictii pentru toate datele de test, comparate cu valorile reale
# for i in range(test_size):
#     print(f"Varsta prezisa: {predictii[i]}  Varsta reala: {y_test[i]}")


print('\nO parte din predictiile facute, comparate cu valorile reale:')
for i in range(15):
    print(f"Varsta prezisa: {predictii[i]}  Varsta reala: {y_test[i]}")
  
# EROARE  
MSE = mean_squared_error(predictii,y_test)
print('\nRMSE: ', str(sqrt(MSE)))    
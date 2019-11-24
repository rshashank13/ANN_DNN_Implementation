from neural_net import nn, activations
from sklearn.model_selection import train_test_split 
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import numpy as np

Datset = datasets.load_iris()

X = Datset.data
y = Datset.target

oneHotEncoder = OneHotEncoder()
oneHotEncoder.fit(y.reshape(-1,1))
y = oneHotEncoder.transform(y.reshape(-1,1)).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

my_nn = nn()

arch_dict = {
            'name' : 'iris_classifier',
            'input_dim': X.shape,
            'n_layers': 2,
            'layer_dims':[10, y.shape[1]],
            'layer_activations':[activations.relu, activations.softmax]
        }
hyper_dict = {
            'n_epoch': 2000,
            'learning_rate': 0.001,
            'batch_size': 10
        }

my_nn.set_architecture(arch_dict)
my_nn.set_hyperparams(hyper_dict)

costs = my_nn.fit(X_train, y_train)

import matplotlib.pyplot as plt

plt.plot(costs)


Y_pred = my_nn.predict(X_train, 10)
y_hat_train = np.argmax(Y_pred, axis=1)
y_train_argmax = np.argmax(y_train, axis=1)
train_accuracy = np.sum(y_hat_train==y_train_argmax)/ len(y_train)

Y_pred = my_nn.predict(X_test, 10)
y_hat_test = np.argmax(Y_pred, axis=1)
y_test_argmax = np.argmax(y_test, axis=1)
test_accuracy = np.sum(y_hat_test==y_test_argmax)/ len(y_test)

import InputLayer, FullyConnectedLayer, LinearLayer, ReLULayer, LogisticSigmoidLayer, SoftmaxLayer, TanhLayer
import pandas as pd
import numpy as np
import math
import SquaredError, LogLoss, CrossEntropy
import matplotlib.pyplot as plt

def accuracy(Y, Yhat):
    pred = np.argmax(Yhat, axis=1)

    Y_ind = np.argmax(Y, axis=1)
    return np.mean(pred == Y_ind.reshape(-1,1))

def shuffle_and_partition(X_train, y_train, batch_size):
    permutation = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]
    
    mini_batches = [
        (X_train_shuffled[i:i + batch_size], y_train_shuffled[i:i + batch_size])
        for i in range(0, X_train.shape[0], batch_size)
    ]
    return mini_batches

train_df = pd.read_csv('./mnist_train_100.csv', header=None)
val_df = pd.read_csv('./mnist_valid_10.csv',header=None)

Y_train = train_df[0].to_numpy()
X_train = train_df.drop(columns = [0]).to_numpy()

Y_val = val_df[0].to_numpy()
X_val = val_df.drop(columns = [0]).to_numpy()

#for cross entropy - one hot!
n = len(np.unique(Y_train))
Y_train = np.eye(n)[Y_train]
Y_val = np.eye(n)[Y_val]


m = X_train.shape[1]
learning_rate = 1e-2
adam_args = {'p1':0.9, 'p2':0.999,'lr': learning_rate}

L1 = InputLayer.InputLayer(X_train)
L2 = FullyConnectedLayer.FullyConnectedLayer(sizeIn = m, sizeOut = n, optimizer = 'ADAM', args = adam_args)
L2.setWeights( np.random.uniform(- np.sqrt(6/(m+n)), np.sqrt(6/(m+n)),size = (m,n)))
L2.setBiases( np.random.uniform(- np.sqrt(6/(m+n)), np.sqrt(6/(m+n)),size = n))
L3 = SoftmaxLayer.SoftmaxLayer()
L4 = CrossEntropy.CrossEntropy()

layers = [L1, L2, L3, L4]


log = {"train_loss":[],"val_loss":[]}
EPOCHS = 5000
batch_size = X_train.shape[0]
best_val_loss = np.inf
patience_counter = 0
patience = 2
for epoch in range(EPOCHS):
    adam_args['epoch'] = epoch
    print('Epoch {}:'.format(epoch+1)) 
    #mini_batches = shuffle_and_partition(X_train, Y_train, batch_size)
    h=X_train
    for i in range(len(layers)-1):
        h = layers[i].forward(h) 
    Yhat_train = h
    MSE_train = layers[-1].eval(Y_train,Yhat_train)
    log['train_loss'].append(MSE_train)
    print('\tTrain: \tAverage Loss: {}'.format(MSE_train))
    grad = layers[-1].gradient(Y_train,Yhat_train)
    for i in range(len(layers)-2,0,-1):
        newgrad = layers[i].backward(grad)
        if(isinstance(layers[i],FullyConnectedLayer.FullyConnectedLayer)): 
            layers[i].updateWeights(grad,learning_rate)
        grad = newgrad
    
    h=X_val
    for i in range(len(layers)-1):
        h = layers[i].forward(h) 
    Yhat_val = h
    MSE_val = layers[-1].eval(Y_val,Yhat_val)
    log['val_loss'].append(MSE_val)
    print('\tTest: \tAverage Loss: {}'.format(MSE_val))

    if MSE_val < best_val_loss:
        best_val_loss = MSE_val
        patience_counter = 0 
    else:
        patience_counter += 1 

    if patience_counter > patience:
        print("Stopping early due to no improvement in validation loss")
        break
    if(epoch > 0 and (abs(log['train_loss'][-2] - log['train_loss'][-1]) < math.pow(10,-6))):
        print("Convergence reached at epoch:", epoch)
        break

print("Train Accuracy: ", accuracy(Y_train, Yhat_train))
print("Val Accuracy: ", accuracy(Y_val, Yhat_val))

x = list(range(epoch+1))
plt.plot(x, log['train_loss'], color="purple", label='Training MSE')
plt.plot(x, log['val_loss'], color="blue",  label='Validation MSE')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
title = 'Gradient descent with ADAM'
plt.title(title)
plt.show()
    


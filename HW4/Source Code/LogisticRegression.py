import InputLayer, FullyConnectedLayer, LinearLayer, ReLULayer, LogisticSigmoidLayer, SoftmaxLayer, TanhLayer
import pandas as pd
import numpy as np
import math
import SquaredError, LogLoss, CrossEntropy
import matplotlib.pyplot as plt

def accuracy(Y, Yhat):
    Y = Y.reshape(Yhat.shape)
    Yhat[Yhat >= 0.5] = 1
    Yhat[Yhat < 0.5] = 0
    return np.mean(Yhat == Y)

df = pd.read_csv('./KidCreative.csv', index_col='Obs No.')


#Shuffle
data = df.sample(frac=1,random_state=0).reset_index(drop=True)
# Calculate the split index
split_index = int(len(data) * 2 / 3)
# Split the data 
train_data = data[:split_index]
val_data = data[split_index:]

Y_train = train_data['Buy'].to_numpy()
X_train = train_data.drop(columns = ['Buy']).to_numpy()

Y_val = val_data['Buy'].to_numpy()
X_val = val_data.drop(columns = ['Buy']).to_numpy()


L1 = InputLayer.InputLayer(X_train)
L2 = FullyConnectedLayer.FullyConnectedLayer(X_train.shape[1],1)
L3 = LogisticSigmoidLayer.LogisticSigmoidLayer()
L4 = LogLoss.LogLoss()

layers = [L1, L2, L3, L4]

learning_rate = 1e-4
EPOCHS = 100000

log = {"train_loss":[],"val_loss":[]}

for epoch in range(EPOCHS):
    print('Epoch {}:'.format(epoch+1)) 
    h=X_train
    for i in range(len(layers)-1):
        h = layers[i].forward(h) 
    Yhat_train = h
    loss_train = layers[-1].eval(Y_train,Yhat_train)
    log['train_loss'].append(loss_train)

    grad = layers[-1].gradient(Y_train,Yhat_train)
    for i in range(len(layers)-2,0,-1):
        newgrad = layers[i].backward(grad)
        if(isinstance(layers[i],FullyConnectedLayer.FullyConnectedLayer)): 
            layers[i].updateWeights(grad,learning_rate)
        grad = newgrad
    
    print('\tTrain: \tAverage Loss: {}'.format(loss_train))
    h=X_val
    for i in range(len(layers)-1):
        h = layers[i].forward(h) 
    Yhat_val = h
    loss_val = layers[-1].eval(Y_val,Yhat_val)
    log['val_loss'].append(loss_val)
    print('\tTest: \tAverage Loss: {}'.format(loss_val))
    if(epoch > 0 and (abs(log['train_loss'][-2] - log['train_loss'][-1]) < math.pow(10,-10))):
        print("Convergence reached at epoch:", epoch)
        break

print("Train Accuracy: ", accuracy(Y_train,Yhat_train))
print("Validation Accuracy: ", accuracy(Y_val,Yhat_val))

x = list(range(epoch+1))
plt.plot(x, log['train_loss'], color="purple", label='Training LogLoss')
plt.plot(x, log['val_loss'], color="blue",  label='Validation LogLoss')
plt.xlabel("Epoch")
plt.ylabel("LogLoss")
plt.legend()
plt.show()
    



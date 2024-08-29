import InputLayer, FullyConnectedLayer, LinearLayer, ReLULayer, LogisticSigmoidLayer, SoftmaxLayer, TanhLayer
import pandas as pd
import numpy as np
import math
import SquaredError, LogLoss, CrossEntropy
import matplotlib.pyplot as plt


def smape(y, yhat):
    y = y.reshape(yhat.shape)
    return np.mean(np.abs(yhat - y) / (np.abs(yhat) + np.abs(y)))


df = pd.read_csv('./medical.csv')
#Shuffle
data = df.sample(frac=1,random_state=0).reset_index(drop=True)
# Calculate the split index
split_index = int(len(data) * 2 / 3)
# Split the data 
train_data = data[:split_index]
val_data = data[split_index:]

Y_train = train_data['charges'].to_numpy()
X_train = train_data.drop(columns = ['charges']).to_numpy()

Y_val = val_data['charges'].to_numpy()
X_val = val_data.drop(columns = ['charges']).to_numpy()


L1 = InputLayer.InputLayer(X_train)
L2 = FullyConnectedLayer.FullyConnectedLayer(X_train.shape[1],1)
L3 = SquaredError.SquaredError()

layers = [L1, L2, L3]

learning_rate = 1e-4
EPOCHS = 100000

log = {"train_loss":[],"val_loss":[]}

for epoch in range(EPOCHS):
    print('Epoch {}:'.format(epoch+1)) 
    h=X_train
    for i in range(len(layers)-1):
        h = layers[i].forward(h) 
    Yhat_train = h
    MSE_train = layers[-1].eval(Y_train,Yhat_train)
    log['train_loss'].append(MSE_train)

    grad = layers[-1].gradient(Y_train,Yhat_train)
    for i in range(len(layers)-2,0,-1):
        newgrad = layers[i].backward(grad)
        if(isinstance(layers[i],FullyConnectedLayer.FullyConnectedLayer)): 
            layers[i].updateWeights(grad,learning_rate)
        grad = newgrad
    
    print('\tTrain: \tAverage Loss: {}'.format(MSE_train))
    h=X_val
    for i in range(len(layers)-1):
        h = layers[i].forward(h) 
    Yhat_val = h
    MSE_val = layers[-1].eval(Y_val,Yhat_val)
    log['val_loss'].append(MSE_val)
    print('\tTest: \tAverage Loss: {}'.format(MSE_val))
    if(epoch > 0 and (abs(log['train_loss'][-2] - log['train_loss'][-1]) < math.pow(10,-10))):
        print("Convergence reached at epoch:", epoch)
        break

rmse_train = np.sqrt(log['train_loss'][-1])
rmse_val = np.sqrt(log['val_loss'][-1])

print("Final RMSE for Train Data: ", rmse_train)
print("Final RMSE for Validation Data: ", rmse_val)

smape_train = smape(Y_train, Yhat_train)
smape_val = smape(Y_val, Yhat_val)

print("Final SMAPE for Train Data: ", smape_train)
print("Final SMAPE for Validation Data: ", smape_val)

x = list(range(epoch+1))
plt.plot(x, log['train_loss'], color="purple", label='Training MSE')
plt.plot(x, log['val_loss'], color="blue",  label='Validation MSE')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()
    


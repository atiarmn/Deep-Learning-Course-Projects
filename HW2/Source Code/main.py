import InputLayer, FullyConnectedLayer, LinearLayer, ReLULayer, LogisticSigmoidLayer, SoftmaxLayer, TanhLayer
import pandas as pd
import numpy as np
import SquaredError, LogLoss, CrossEntropy


df = pd.read_csv('./KidCreative.csv', index_col='Obs No.')
Y = df['Buy'].to_numpy()
X = df.drop(columns = ['Buy']).to_numpy()

L1 = InputLayer.InputLayer(X)
try:
    L2 = FullyConnectedLayer.FullyConnectedLayer(X.shape[1],1)
except:
    #for single observation! for example : X = df.iloc[0].to_numpy()
    L2 = FullyConnectedLayer.FullyConnectedLayer(X.shape[0],1)

L3 = LogisticSigmoidLayer.LogisticSigmoidLayer()
L4 = LogLoss.LogLoss()

layers = [L1, L2, L3, L4]

h=X
for i in range(len(layers)-1):
    h = layers[i].forward(h) 
Yhat = h

#for cross entropy - one hot!
#n_values = Yhat.shape[1]
#Y = np.eye(n_values)[Y]

print(layers[-1].eval(Y,Yhat))
grad = layers[-1].gradient(Y,Yhat)
print("Mean over observations in layer",layers[-1], ": \n", grad.mean(axis = 0))
for i in range(len(layers)-2,0,-1):
    grad = layers[i].backward(grad)
    print("Mean over observations in layer",layers[i], ": \n", grad.mean(axis = 0))



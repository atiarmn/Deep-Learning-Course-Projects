import InputLayer, FullyConnectedLayer, LinearLayer, ReLULayer, LogisticSigmoidLayer, SoftmaxLayer, TanhLayer
import pandas as pd
import numpy as np

df = pd.read_csv('./KidCreative.csv', index_col='Obs No.')
y = df['Buy'].to_numpy()
X = df.drop(columns = ['Buy']).to_numpy()

L1 = InputLayer.InputLayer(X)
L2 = FullyConnectedLayer.FullyConnectedLayer(X.shape[1],1) 
L3 = LogisticSigmoidLayer.LogisticSigmoidLayer()
layers = [L1, L2, L3]

h=X
for i in range(len(layers)):
    h = layers[i].forward(h) 
Yhat = h
print(Yhat)
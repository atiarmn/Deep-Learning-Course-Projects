import InputLayer, FullyConnectedLayer, LinearLayer, ReLULayer, LogisticSigmoidLayer, SoftmaxLayer, TanhLayer
import pandas as pd
import numpy as np
import math
import SquaredError, LogLoss, CrossEntropy
import matplotlib.pyplot as plt
import ConvolutionalLayer, MaxPoolLayer, FlatteningLayer


L1 = ConvolutionalLayer.ConvolutionalLayer(3,3)
L2 = MaxPoolLayer.MaxPoolLayer(3,3)
L3 = FlatteningLayer.FlatteningLayer()
L4 = FullyConnectedLayer.FullyConnectedLayer(4,1)
L4.setWeights(np.array([[-1],[3],[0],[-1]]))
L4.setBiases([0])
L5 = LinearLayer.LinearLayer()
L6 = SquaredError.SquaredError()

layers = [L1, L2, L3, L4, L5, L6]
learning_rate = 1e-4

X = np.array([[1,1,0,1,0,0,1,1],[1,1,1,1,0,0,1,0],[0,0,1,1,0,1,0,1],[1,1,1,0,1,1,1,0],[1,1,1,1,1,0,1,1],[0,0,0,0,0,0,0,0],[0,1,1,1,1,0,0,1],[1,0,1,0,0,1,0,1]])
L1.setKernel(np.array([[2,-1,2],[2,-1,0],[1,0,2]]))
y = np.array([5])

h=X
for i in range(len(layers)-1):
    print("input of layer ", layers[i], " is : \n ", h)
    h = layers[i].forward(h) 
    print("output of layer ", layers[i], " is : \n ", h)
Yhat_train = h
print("Yhat is: ", Yhat_train)
loss_train = layers[-1].eval(y,Yhat_train)
print("loss is: ", loss_train)
grad = layers[-1].gradient(y,Yhat_train)
for i in range(len(layers)-2,-1,-1):
    print("gradIn of layer ", layers[i], " is : \n ", grad)
    newgrad = layers[i].backward(grad)
    print("grad out of layer ", layers[i], " is : \n ", newgrad)
    if(isinstance(layers[i],FullyConnectedLayer.FullyConnectedLayer) or isinstance(layers[i],ConvolutionalLayer.ConvolutionalLayer)): 
        print("update weights of layer ", layers[i])
        layers[i].updateWeights(grad,learning_rate)
        print("updated weight: ")
        if(isinstance(layers[i],FullyConnectedLayer.FullyConnectedLayer)):
            print(layers[i].getWeights())
        if(isinstance(layers[i],ConvolutionalLayer.ConvolutionalLayer)):
            print(layers[i].getKernel())
    grad = newgrad

print("final grad:" , grad)
# learning_rate = 1e-4
# EPOCHS = 100000

# log = {"train_loss":[],"val_loss":[]}

# for epoch in range(EPOCHS):
#     print('Epoch {}:'.format(epoch+1)) 
#     h=X_train
#     for i in range(len(layers)-1):
#         h = layers[i].forward(h) 
#     Yhat_train = h
#     loss_train = layers[-1].eval(Y_train,Yhat_train)
#     log['train_loss'].append(loss_train)

#     grad = layers[-1].gradient(Y_train,Yhat_train)
#     for i in range(len(layers)-2,0,-1):
#         newgrad = layers[i].backward(grad)
#         if(isinstance(layers[i],FullyConnectedLayer.FullyConnectedLayer)): 
#             layers[i].updateWeights(grad,learning_rate)
#         grad = newgrad
    
#     print('\tTrain: \tAverage Loss: {}'.format(loss_train))
#     h=X_val
#     for i in range(len(layers)-1):
#         h = layers[i].forward(h) 
#     Yhat_val = h
#     loss_val = layers[-1].eval(Y_val,Yhat_val)
#     log['val_loss'].append(loss_val)
#     print('\tTest: \tAverage Loss: {}'.format(loss_val))
#     if(epoch > 0 and (abs(log['train_loss'][-2] - log['train_loss'][-1]) < math.pow(10,-10))):
#         print("Convergence reached at epoch:", epoch)
#         break

# print("Train Accuracy: ", accuracy(Y_train,Yhat_train))
# print("Validation Accuracy: ", accuracy(Y_val,Yhat_val))

# x = list(range(epoch+1))
# plt.plot(x, log['train_loss'], color="purple", label='Training LogLoss')
# plt.plot(x, log['val_loss'], color="blue",  label='Validation LogLoss')
# plt.xlabel("Epoch")
# plt.ylabel("LogLoss")
# plt.legend()
# plt.show()
    



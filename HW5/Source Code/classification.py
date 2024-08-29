import InputLayer, FullyConnectedLayer, LinearLayer, ReLULayer, LogisticSigmoidLayer, SoftmaxLayer, TanhLayer
import pandas as pd
import numpy as np
import SquaredError, LogLoss, CrossEntropy
import matplotlib.pyplot as plt
import ConvolutionalLayer, MaxPoolLayer, FlatteningLayer
import os
from PIL import Image



def plot_kernel(kernel, title):
    plt.imshow(kernel, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()


def accuracy(Y, Yhat):
    Y = Y.reshape(Yhat.shape)
    Yhat[Yhat >= 0.5] = 1
    Yhat[Yhat < 0.5] = 0
    return np.mean(Yhat == Y)


dataset_directory = './yalefaces'

X_train = []
y_train = []

X_test = []
y_test = []


for filename in os.listdir(dataset_directory):
    if(filename.__contains__("Read") or filename.__contains__("DS_Store")):
        continue
    filepath = os.path.join(dataset_directory, filename)
    img = Image.open(filepath)
    img = img.resize((40, 40))
    image = np.array(img)
    label = int(filename.split('.')[0].lstrip('subject')) - 2
    if(label not in y_train):
        X_train.append(image)
        y_train.append(label)
    else:
        X_test.append(image)
        y_test.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


n = len(np.unique(y_train))
y_train = np.eye(n)[y_train]


L1 = InputLayer.InputLayer(X_train)
L2 = ConvolutionalLayer.ConvolutionalLayer(9,9)
L3 = MaxPoolLayer.MaxPoolLayer(4,4)
L4 = FlatteningLayer.FlatteningLayer()
L5 = FullyConnectedLayer.FullyConnectedLayer(64,n)
L6 = SoftmaxLayer.SoftmaxLayer()
L7 = CrossEntropy.CrossEntropy()

layers = [L1, L2, L3, L4, L5, L6, L7]

n = len(np.unique(y_test))
y_test = np.eye(n)[y_test]

learning_rate = 1e-2
EPOCHS = 5000


log = {"train_loss":[]}

init_kernel = L2.getKernel()
for epoch in range(EPOCHS):
    print('Epoch {}:'.format(epoch+1)) 
    h=X_train
    for i in range(len(layers)-1):
        h = layers[i].forward(h) 
    Yhat_train = h
    loss_train = layers[-1].eval(y_train,Yhat_train)
    log['train_loss'].append(loss_train)
    grad = layers[-1].gradient(y_train,Yhat_train)
    for i in range(len(layers)-2,0,-1):
        newgrad = layers[i].backward(grad)
        if(isinstance(layers[i],FullyConnectedLayer.FullyConnectedLayer) or isinstance(layers[i],ConvolutionalLayer.ConvolutionalLayer)): 
            layers[i].updateWeights(grad,learning_rate)
        grad = newgrad
    print('\tTrain: \tAverage Loss: {}'.format(loss_train))
    if(epoch > 0 and (abs(log['train_loss'][-2] - log['train_loss'][-1]) < 1e-10)):
        print("Convergence reached at epoch:", epoch)
        break
final_kernel = L2.getKernel()

h=X_test
for i in range(len(layers)-1):
    h = layers[i].forward(h) 
Yhat_test = h
loss_test = layers[-1].eval(y_test,Yhat_test)

print("Train Accuracy: ", accuracy(y_train,Yhat_train))
print("Test Accuracy: ", accuracy(y_test,Yhat_test))

x = list(range(epoch+1))
plt.plot(x, log['train_loss'])
plt.xlabel("Epoch")
plt.ylabel("CrossLoss")
plt.show()

plot_kernel(init_kernel, 'Initial Kernel')


plot_kernel(final_kernel, 'Final Kernel')

    

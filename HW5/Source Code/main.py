import InputLayer, FullyConnectedLayer, LinearLayer, ReLULayer, LogisticSigmoidLayer, SoftmaxLayer, TanhLayer
import numpy as np
import SquaredError, LogLoss, CrossEntropy
import matplotlib.pyplot as plt
import ConvolutionalLayer, MaxPoolLayer, FlatteningLayer


def create_img(image_size):
    vertical_stripe_image = np.zeros(image_size)
    vertical_stripe_position = 38
    vertical_stripe_image[:, vertical_stripe_position] = 1
    
    plt.imshow(vertical_stripe_image, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title("vertical stripe image")
    plt.show()

    horizontal_stripe_image = np.zeros(image_size)
    horizontal_stripe_position = 1
    horizontal_stripe_image[horizontal_stripe_position, :] = 1

    plt.imshow(horizontal_stripe_image, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title("horizontal stripe image")
    plt.show()
    
    return np.array([vertical_stripe_image,horizontal_stripe_image])

def plot_kernel(kernel, title):
    plt.imshow(kernel, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()

L1 = ConvolutionalLayer.ConvolutionalLayer(9,9)
L2 = MaxPoolLayer.MaxPoolLayer(4,4)
L3 = FlatteningLayer.FlatteningLayer()
L4 = FullyConnectedLayer.FullyConnectedLayer(64,1)
L5 = LogisticSigmoidLayer.LogisticSigmoidLayer()
L6 = LogLoss.LogLoss()

layers = [L1, L2, L3, L4, L5, L6]

X = create_img(image_size=(40,40))
y = np.array([[0],[1]])


learning_rate = 1e-2
EPOCHS = 2000

log = {"loss":[]}

init_kernel = L1.getKernel()
for epoch in range(EPOCHS):
    print('Epoch {}:'.format(epoch+1)) 
    h=X
    for i in range(len(layers)-1):
        h = layers[i].forward(h) 
    Yhat_train = h
    loss_train = layers[-1].eval(y,Yhat_train)
    log['loss'].append(loss_train)
    grad = layers[-1].gradient(y,Yhat_train)
    for i in range(len(layers)-2,-1,-1):
        newgrad = layers[i].backward(grad)
        if(isinstance(layers[i],FullyConnectedLayer.FullyConnectedLayer) or isinstance(layers[i],ConvolutionalLayer.ConvolutionalLayer)): 
            layers[i].updateWeights(grad,learning_rate)
        grad = newgrad
    print('\tAverage Loss: {}'.format(loss_train))
    
final_kernel = L1.getKernel()
x = list(range(epoch+1))
plt.plot(x, log['loss'])
plt.xlabel("Epoch")
plt.ylabel("LogLoss")
plt.show()

plot_kernel(init_kernel, 'Initial Kernel')


plot_kernel(final_kernel, 'Final Kernel')

    

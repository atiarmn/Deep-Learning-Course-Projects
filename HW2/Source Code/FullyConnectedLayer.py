from Layer import Layer
import numpy as np
class FullyConnectedLayer(Layer):
    def __init__ (self , sizeIn , sizeOut):
        super().__init__()
        self.__weight = np.random.uniform(-1e-4, 1e-4,size = (sizeIn,sizeOut))
        self.__bias = np.random.uniform(-1e-4, 1e-4,size = sizeOut)
    def getWeights(self): 
        return self.__weight
    def setWeights(self , weights):
        self.__weight = weights
    def getBiases(self):
        return self.__bias
    def setBiases(self , biases):
        self.__bias = biases
    def forward(self ,dataIn):
        self.setPrevIn(dataIn)
        output = np.dot(dataIn,self.__weight) + self.__bias
        self.setPrevOut(output)
        return output
    def gradient(self):
        return self.getWeights().T
    def backward( self , gradIn ): 
        return gradIn @ self.gradient()
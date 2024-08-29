from Layer import Layer
import numpy as np
class InputLayer(Layer):
    def __init__ (self ,dataIn): 
        self.meanX = np.mean(dataIn, axis=0)
        self.stdX = np.std(dataIn, axis=0)
        self.stdX[self.stdX == 0] = 1
        self.input = dataIn
        self.output = []
    def forward(self ,dataIn): 
        normalized_data = (dataIn - self.meanX) / self.stdX
        self.output = normalized_data
        return normalized_data
    def gradient(self): 
        pass
    def backward( self , gradIn ): 
        pass
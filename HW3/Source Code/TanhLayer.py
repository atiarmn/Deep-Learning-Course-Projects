from Layer import Layer
import numpy as np
class TanhLayer(Layer): 
    def __init__(self):
        super().__init__()
    def forward(self ,dataIn): 
        self.setPrevIn(dataIn)
        tanh_value = np.matrix((np.exp(dataIn)-np.exp(-dataIn))/(np.exp(dataIn)+np.exp(-dataIn)), dtype = float)
        self.setPrevOut(tanh_value)
        return tanh_value
    def gradient(self): 
        tanh_derivative = 1 - np.power(self.getPrevOut(), 2)
        return tanh_derivative
    
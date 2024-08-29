from Layer import Layer
import numpy as np
class ReLULayer(Layer): 
    def __init__(self):
        super().__init__()
    def forward(self ,dataIn): 
        self.setPrevIn(dataIn)
        relu_value = np.matrix(dataIn, dtype = float)
        relu_value[relu_value < 0] = 0 
        self.setPrevOut(relu_value)
        return relu_value
    def gradient(self): 
        relu_derivative = np.matrix(self.getPrevIn(), dtype = float)
        relu_derivative[relu_derivative >= 0] = 1
        relu_derivative[relu_derivative < 0] = 0
        return relu_derivative
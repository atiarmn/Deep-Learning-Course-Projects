from Layer import Layer
import numpy as np
class LinearLayer(Layer): 
    def __init__(self):
        super().__init__()
    def forward(self ,dataIn): 
        self.setPrevIn(dataIn)
        identical_value = np.matrix(dataIn, dtype=float)
        self.setPrevOut(identical_value)
        return identical_value
    def gradient(self): 
        temp = np.matrix(self.getPrevIn(), dtype=float)
        identical_derivative = np.matrix(np.full(np.shape(temp), 1.))
        return identical_derivative
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
        pass
    def backward( self , gradIn ): 
        pass
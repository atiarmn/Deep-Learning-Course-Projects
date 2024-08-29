from Layer import Layer
import numpy as np
class LogisticSigmoidLayer(Layer): 
    def __init__(self):
        super().__init__()
    def forward(self ,dataIn): 
        self.setPrevIn(dataIn)
        sigmoid_value = np.matrix(1/(1+np.exp(-dataIn)),dtype = float)
        self.setPrevOut(sigmoid_value)
        return sigmoid_value
    def gradient(self): 
        pass
    def backward( self , gradIn ): 
        pass
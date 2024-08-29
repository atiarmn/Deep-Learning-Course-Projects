from Layer import Layer
import numpy as np
class SoftmaxLayer(Layer): 
    def __init__(self):
        super().__init__()
    def forward(self ,dataIn): 
        self.setPrevIn(dataIn)
        shifted_matrix = dataIn - dataIn.max()
        expo = np.exp(shifted_matrix)
        expo_sum = np.sum(np.exp(shifted_matrix))
        softmax_value = np.matrix(expo / expo_sum,dtype = float)
        self.setPrevOut(softmax_value)
        return softmax_value
    def gradient(self): 
        pass
    def backward( self , gradIn ): 
        pass
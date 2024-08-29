from Layer import Layer
import numpy as np
class FlatteningLayer(Layer):
    def __init__ (self):
        super().__init__()
    def forward(self ,dataIn):
        self.setPrevIn(dataIn)
        out = dataIn.reshape(dataIn.shape[0], -1) if dataIn.ndim > 2 else dataIn.reshape(1,-1)
        self.setPrevOut(out)
        return out
    def gradient(self):
        pass
    def backward( self , gradIn ): 
        return gradIn.reshape(self.getPrevIn().shape)
    
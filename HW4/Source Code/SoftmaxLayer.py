from Layer import Layer
import numpy as np
import torch
class SoftmaxLayer(Layer): 
    def __init__(self):
        super().__init__()
    def forward(self ,dataIn): 
        self.setPrevIn(dataIn)
        try:
            shifted_matrix = dataIn - dataIn.max(axis = 1).reshape(-1,1)
        except:
            # for one observation
            shifted_matrix = dataIn
        
        expo = np.exp(shifted_matrix)
        expo_sum = np.sum(np.exp(shifted_matrix),axis = 1).reshape(-1,1)
        softmax_value = np.matrix(expo / expo_sum,dtype = float)
        self.setPrevOut(softmax_value)
        return softmax_value
    def gradient(self): 
        softmax_derivative = []
        for obs in self.getPrevOut():
            flat = np.asarray(obs).reshape(-1)
            drev = np.diag(flat) - np.transpose(obs)*obs
            softmax_derivative.append(drev)
        softmax_derivative = np.array(softmax_derivative)
        return torch.tensor(softmax_derivative)
    def backward( self , gradIn ): 
        return np.einsum('...i,...ij',gradIn,self.gradient())
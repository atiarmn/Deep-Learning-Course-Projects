import numpy as np
class CrossEntropy():
    def eval(self ,Y, Yhat):
        eps = np.finfo(Yhat.dtype).eps
        if(isinstance(Yhat,np.matrix)):
            Yhat = np.asarray(Yhat).reshape(-1)
        if(Y.shape != Yhat.shape):
            Y = Y.reshape(Yhat.shape)
        temp = -Y * np.log(Yhat + eps)
        losses = temp
        if(len(temp.shape) > 1):
            losses = np.sum(temp, axis = 1)
        return np.mean(losses)
    
    def gradient(self ,Y, Yhat):
        eps = np.finfo(Yhat.dtype).eps
        if(Y.shape != Yhat.shape):
            Y = Y.reshape(Yhat.shape)
        return - Y / (Yhat + eps)
    




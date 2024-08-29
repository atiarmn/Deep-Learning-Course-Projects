import numpy as np
class LogLoss():
    def eval(self ,Y, Yhat): 
        eps = np.finfo(Yhat.dtype).eps
        if(isinstance(Yhat,np.matrix)):
            Yhat = np.asarray(Yhat).reshape(-1,1)
        temp = -(Y * np.log(Yhat + eps) + ((1 - Y) * np.log(1 - Yhat + eps)))
        return np.mean(temp)
    def gradient(self ,Y, Yhat):
        eps = np.finfo(Yhat.dtype).eps
        if(Y.shape != Yhat.shape):
            Y = Y.reshape(Yhat.shape)
        return - (Y - Yhat) / (np.multiply(Yhat,(1 - Yhat)) + eps)
    

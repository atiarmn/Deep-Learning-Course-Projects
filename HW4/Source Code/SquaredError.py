import numpy as np
class SquaredError():
    def eval(self ,Y, Yhat): 
        if(isinstance(Yhat,np.matrix)):
            Yhat = np.asarray(Yhat).reshape(-1)
        return np.mean((Y - Yhat)*(Y - Yhat))
    def gradient(self ,Y, Yhat):
        if(Y.shape != Yhat.shape):
            Y = Y.reshape(Yhat.shape)
        return -2 * (Y - Yhat)
from Layer import Layer
import numpy as np
class FullyConnectedLayer(Layer):
    def __init__ (self , sizeIn , sizeOut, optimizer = None, args = None):
        super().__init__()
        self.__weight = np.random.uniform(-1e-4, 1e-4,size = (sizeIn,sizeOut))
        self.__bias = np.random.uniform(-1e-4, 1e-4,size = sizeOut)
        self._optimizer = optimizer
        self._opt_args = args
        self._sw = 0
        self._rw = 0
        self._sb = 0
        self._rb = 0
        self._delta = 10**(-8)
    def getWeights(self): 
        return self.__weight
    def setWeights(self , weights):
        self.__weight = weights
    def getBiases(self):
        return self.__bias
    def setBiases(self , biases):
        self.__bias = biases
    def forward(self ,dataIn):
        self.setPrevIn(dataIn)
        output = np.dot(dataIn,self.__weight) + self.__bias
        self.setPrevOut(output)
        return output
    def gradient(self):
        return self.getWeights().T
    def backward( self , gradIn ): 
        return gradIn @ self.gradient()
    def updateWeights(self, gradIn, eta):
        dJdb = np.sum(gradIn, axis = 0)/gradIn.shape[0]
        dJdW = (self.getPrevIn().T @ gradIn)/gradIn.shape[0]

        if(self._optimizer == 'ADAM'):
            self._sw = self._opt_args['p1']* self._sw + (1 - self._opt_args['p1']) * dJdW
            self._rw = self._opt_args['p2']* self._rw + (1 - self._opt_args['p2']) * (dJdW*dJdW)
            self._sb = self._opt_args['p1']* self._sb + (1 - self._opt_args['p1']) * dJdb
            self._rb = self._opt_args['p2']* self._rb + (1 - self._opt_args['p2']) * (dJdb*dJdb)
            t = self._opt_args['epoch'] + 1
            adam_w = (self._sw/(1 - self._opt_args['p1']**t))/(np.sqrt((self._rw/(1 - self._opt_args['p2']**t))) + self._delta)
            self.setWeights(self.getWeights() - (self._opt_args['lr'] * adam_w))
            adam_b = (self._sb/(1 - self._opt_args['p1']**t))/(np.sqrt((self._rb/(1 - self._opt_args['p2']**t))) + self._delta)
            self.setBiases(self.getBiases() - (self._opt_args['lr'] * adam_b))
        else:   
            self.setWeights(self.getWeights() - eta*dJdW)
            self.setBiases(self.getBiases() - eta*dJdb)
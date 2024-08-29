from abc import ABC, abstractmethod
import numpy as np
class Layer(ABC):
  def __init__(self):
    self.__prevIn = []
    self.__prevOut = []
  def setPrevIn(self,dataIn):
    self.__prevIn = dataIn
  def setPrevOut(self, out):
    self.__prevOut = out
  def getPrevIn(self):
    return self.__prevIn
  def getPrevOut(self):
    return self.__prevOut
  @abstractmethod
  def forward(self,dataIn):
    pass
  @abstractmethod
  def gradient(self):
    pass
  def backward(self,gradIn):
    prevGrad = np.asarray(gradIn)
    selfGrad = np.asarray(self.gradient())
    return prevGrad * selfGrad
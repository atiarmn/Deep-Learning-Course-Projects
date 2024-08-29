import numpy as np
from RNN import RNN
import pandas as pd
from DataGen import DataGenerator

data_generator = DataGenerator('./words.txt')
rnn = RNN(hidden_size=100,data_generator=data_generator, sequence_length=25, learning_rate=1e-3)
rnn.train()
print("\n\nTesting: ")
print(rnn.predict("searc"))
print(rnn.predict("th"))
print(rnn.predict("hi"))
print(rnn.predict("c"))
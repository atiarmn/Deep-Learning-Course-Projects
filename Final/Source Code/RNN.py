import numpy as np
class RNN:
    def __init__(self, hidden_size, data_generator, sequence_length, learning_rate):
        self.hidden_size = hidden_size
        self.data_generator = data_generator
        self.vocab_size = self.data_generator.vocab_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.X = None

        self.Wax = np.random.uniform(-np.sqrt(1. / self.vocab_size), np.sqrt(1. / self.vocab_size), (hidden_size, self.vocab_size))
        self.Waa = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (hidden_size, hidden_size))
        self.Wya = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (self.vocab_size, hidden_size))
        self.ba = np.zeros((hidden_size, 1))  
        self.by = np.zeros((self.vocab_size, 1))
        
        self.dWax, self.dWaa, self.dWya = np.zeros_like(self.Wax), np.zeros_like(self.Waa), np.zeros_like(self.Wya)
        self.dba, self.dby = np.zeros_like(self.ba), np.zeros_like(self.by)
        
        self.mWax = np.zeros_like(self.Wax)
        self.vWax = np.zeros_like(self.Wax)
        self.mWaa = np.zeros_like(self.Waa)
        self.vWaa = np.zeros_like(self.Waa)
        self.mWya = np.zeros_like(self.Wya)
        self.vWya = np.zeros_like(self.Wya)
        self.mba = np.zeros_like(self.ba)
        self.vba = np.zeros_like(self.ba)
        self.mby = np.zeros_like(self.by)
        self.vby = np.zeros_like(self.by)

    def softmax(self, x):
        x = x - np.max(x)
        p = np.exp(x)
        return p / np.sum(p)

    def forward(self, X, a_prev):
        x, a, y_pred = {}, {}, {}
        self.X = X

        a[-1] = np.copy(a_prev)
        for t in range(len(self.X)): 
            x[t] = np.zeros((self.vocab_size,1)) 
            if (self.X[t] != None):
                x[t][self.X[t]] = 1
            a[t] = np.tanh(np.dot(self.Wax, x[t]) + np.dot(self.Waa, a[t - 1]) + self.ba)
            y_pred[t] = self.softmax(np.dot(self.Wya, a[t]) + self.by)
        return x, a, y_pred 
    
    def backward(self,x, a, y_preds, targets):
        da_next = np.zeros_like(a[0])

        for t in reversed(range(len(self.X))):
            dy_preds = np.copy(y_preds[t])
            dy_preds[targets[t]] -= 1
            da = np.dot(self.Waa.T, da_next) + np.dot(self.Wya.T, dy_preds)
            dtanh = (1 - np.power(a[t], 2))
            da_unactivated = dtanh * da
            self.dba += da_unactivated
            self.dWax += np.dot(da_unactivated, x[t].T)
            self.dWaa += np.dot(da_unactivated, a[t - 1].T)
            da_next = da_unactivated
            self.dWya += np.dot(dy_preds, a[t].T)
            for grad in [self.dWax, self.dWaa, self.dWya, self.dba, self.dby]:
                np.clip(grad, -1, 1, out=grad)
 
    def loss(self, y_preds, targets):
        return sum(-np.log(y_preds[t][targets[t], 0]) for t in range(len(self.X)))
    
    def adamw(self, beta1=0.9, beta2=0.999, epsilon=1e-8, L2_reg=1e-4):
  
        self.mWax = beta1 * self.mWax + (1 - beta1) * self.dWax
        self.vWax = beta2 * self.vWax + (1 - beta2) * np.square(self.dWax)
        m_hat = self.mWax / (1 - beta1)
        v_hat = self.vWax / (1 - beta2)
        self.Wax -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.Wax)

        self.mWaa = beta1 * self.mWaa + (1 - beta1) * self.dWaa
        self.vWaa = beta2 * self.vWaa + (1 - beta2) * np.square(self.dWaa)
        m_hat = self.mWaa / (1 - beta1)
        v_hat = self.vWaa / (1 - beta2)
        self.Waa -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.Waa)

        self.mWya = beta1 * self.mWya + (1 - beta1) * self.dWya
        self.vWya = beta2 * self.vWya + (1 - beta2) * np.square(self.dWya)
        m_hat = self.mWya / (1 - beta1)
        v_hat = self.vWya / (1 - beta2)
        self.Wya -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.Wya)

        self.mba = beta1 * self.mba + (1 - beta1) * self.dba
        self.vba = beta2 * self.vba + (1 - beta2) * np.square(self.dba)
        m_hat = self.mba / (1 - beta1)
        v_hat = self.vba / (1 - beta2)
        self.ba -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.ba)

        self.mby = beta1 * self.mby + (1 - beta1) * self.dby
        self.vby = beta2 * self.vby + (1 - beta2) * np.square(self.dby)
    
    def sample(self):

        x = np.zeros((self.vocab_size, 1))
        a_prev = np.zeros((self.hidden_size, 1))

        indices = []

        idx = -1

        counter = 0
        max_chars = 50
        newline_character = self.data_generator.char_to_idx['\n'] 

        while (idx != newline_character and counter != max_chars):
            a = np.tanh(np.dot(self.Wax, x) + np.dot(self.Waa, a_prev) + self.ba)
            y = self.softmax(np.dot(self.Wya, a) + self.by)
            idx = np.random.choice(list(range(self.vocab_size)), p=y.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            indices.append(idx)
            a_prev = a
            counter += 1

        return indices

        
    def train(self, generated_names=5):

        iter_num = 0
        threshold = 3.46 # stopping criterion for training
        smooth_loss = -np.log(1.0 / self.data_generator.vocab_size) * self.sequence_length

        while (smooth_loss > threshold):
            a_prev = np.zeros((self.hidden_size, 1))
            idx = iter_num % self.vocab_size
            inputs, targets = self.data_generator.generate_example(idx)

            x, a, y_pred  = self.forward(inputs, a_prev)
            self.backward(x, a, y_pred, targets)
            loss = self.loss(y_pred, targets)
            self.adamw()
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            a_prev = a[len(self.X) - 1]

            if iter_num % 500 == 0:
                print("\n\niter :%d, loss:%f\n" % (iter_num, smooth_loss))
                for i in range(generated_names):
                    sample_idx = self.sample()
                    txt = ''.join(self.data_generator.idx_to_char[idx] for idx in sample_idx)
                    txt = txt.title()
                    print ('%s' % (txt, ), end='')
            iter_num += 1
    
    def predict(self, start):
        x = np.zeros((self.vocab_size, 1))
        a_prev = np.zeros((self.hidden_size, 1))

        chars = [ch for ch in start]
        idxes = []
        for i in range(len(chars)):
            idx = self.data_generator.char_to_idx[chars[i]]
            x[idx] = 1
            idxes.append(idx)

        max_chars = 50
        newline_character = self.data_generator.char_to_idx['\n']
        counter = 0
        while (idx != newline_character and counter != max_chars):
            a = np.tanh(np.dot(self.Wax, x) + np.dot(self.Waa, a_prev) + self.ba)
            y_pred = self.softmax(np.dot(self.Wya, a) + self.by)
            idx = np.random.choice(range(self.vocab_size), p=y_pred.ravel())

            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            a_prev = a
            idxes.append(idx)
            counter += 1

        txt = ''.join(self.data_generator.idx_to_char[i] for i in idxes)
        if txt[-1] == '\n':
            txt = txt[:-1]

        return txt

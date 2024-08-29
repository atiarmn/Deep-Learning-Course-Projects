from Layer import Layer
import numpy as np
class MaxPoolLayer(Layer):
    def __init__ (self , poolSize, stride):
        super().__init__()
        self._pool_size = poolSize
        self._stride = stride
        self._mask = None
    def forward(self ,dataIn):
        self.setPrevIn(dataIn)
        
        if dataIn.ndim == 2:
            # Single image - reshape to include batch dimension of 1
            dataIn = dataIn.reshape((1,) + dataIn.shape)

        batch_size, height, width = dataIn.shape

        self._mask = np.zeros_like(dataIn, dtype=bool)

        out_height = (height - self._pool_size) // self._stride + 1
        out_width = (width - self._pool_size) // self._stride + 1

        output = np.zeros((batch_size,out_height, out_width))
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * self._stride
                    h_end = h_start + self._pool_size
                    w_start = w * self._stride
                    w_end = w_start + self._pool_size
                    window = dataIn[b, h_start:h_end, w_start:w_end]
                    output[b, h, w] = np.max(window)
                    max_val_idx = np.unravel_index(np.argmax(window),window.shape)
                    self._mask[b, h_start + max_val_idx[0], w_start + max_val_idx[1]] = True
        
        if batch_size == 1:
            output = output.reshape(out_height, out_width)
            self._mask = self._mask.reshape(height, width)

        self.setPrevOut(output)
        return output
    def gradient(self):
        pass
    def backward(self , gradIn): 
        prevIn = self.getPrevIn()
        mask = self._mask
        if prevIn.ndim == 2:
            # Single image - reshape to include batch dimension of 1
            prevIn = prevIn.reshape((1,) + prevIn.shape)
        
        if gradIn.ndim == 2:
            # Single image - reshape to include batch dimension of 1
            gradIn = gradIn.reshape((1,) + gradIn.shape)

        if mask.ndim == 2:
            # Single image - reshape to include batch dimension of 1
            mask = mask.reshape((1,) + mask.shape)

        batch_size, out_height, out_width = gradIn.shape
        height, width = prevIn.shape[1:]
        output = np.zeros((batch_size, height, width))

        for b in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    y_start = y * self._stride
                    x_start = x * self._stride
                    output[b,y_start:y_start+self._pool_size, x_start:x_start+self._pool_size] = (mask[b,y_start:y_start+self._pool_size, x_start:x_start+self._pool_size] * gradIn[b, y, x])
        if batch_size == 1:
            output = output.reshape(height, width)
        return output
    
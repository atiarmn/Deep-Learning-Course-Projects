from Layer import Layer
import numpy as np
class ConvolutionalLayer(Layer):
    def __init__ (self , kHeight , kWidth):
        super().__init__()
        self._kernel = np.random.uniform(-1e-4, 1e-4,size = (kHeight,kWidth))
    def getKernel(self): 
        return self._kernel
    def setKernel(self , kernel):
        self._kernel = kernel
    def forward(self ,dataIn):
        self.setPrevIn(dataIn)
        cross = self.crossCorrelate2D(dataIn, self._kernel)
        self.setPrevOut(cross)
        return cross
    def gradient(self):
        pass
    def backward( self , gradIn ): 
        pass
    def updateWeights(self, gradIn, eta):
        if gradIn.ndim > 2 :
            dJdK = np.zeros_like(self._kernel)
        
            for i in range(gradIn.shape[0]):  
                dJdK += self.crossCorrelate2D(self.getPrevIn()[i], gradIn[i])
    
            dJdK /= gradIn.shape[0]
        else:
            dJdK = self.crossCorrelate2D(self.getPrevIn(), gradIn)
        
        self._kernel = self._kernel.astype(np.float64)
        self._kernel -= eta * dJdK
    def crossCorrelate2D(self, dataIn, kernel):
        if dataIn.ndim == 2:
            # Single image - reshape to include batch dimension of 1
            dataIn = dataIn.reshape((1,) + dataIn.shape)
            
        kernel_rows, kernel_cols = kernel.shape
        batch_size, input_rows, input_cols = dataIn.shape

        output_rows = input_rows - kernel_rows + 1
        output_cols = input_cols - kernel_cols + 1
        
        output = np.zeros((batch_size,output_rows,output_cols))
        
        # for b in range(batch_size):
        #     for i in range(output_rows):
        #         for j in range(output_cols):
        #             output[b, i, j] = np.sum(dataIn[b, i:i+kernel_rows, j:j+kernel_cols] * kernel)

        # Generate all possible windows for each image in the batch
        windows_shape = (batch_size, output_rows, output_cols, kernel_rows, kernel_cols)
        
        windows_strides = (
            dataIn.strides[0],                       # Move to the next image in the batch
            dataIn.strides[1],              # Move down by 'stride' rows
            dataIn.strides[2],              # Move right by 'stride' columns
            dataIn.strides[1],                       # Move down by one row (within a window)
            dataIn.strides[2]                        # Move right by one column (within a window)
        )

        
        windows = np.lib.stride_tricks.as_strided(
            dataIn,
            shape=windows_shape,
            strides=windows_strides
        )
        
        # Perform element-wise multiplication and sum across the kernel dimensions
        # This effectively applies the kernel to each window
        output = np.einsum('bijxy,xy->bij', windows, kernel)
        
        if batch_size == 1:
            return output[0]
    
        return output
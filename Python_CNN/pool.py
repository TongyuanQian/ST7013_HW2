import numpy as np


class MaxPool:
    '''
    This is a max pooling layer that using an n x n pool
    '''
    def __init__(self, pool_size):
        '''
        Initialize a max pooling layer with dimensions:
        (pool_size, pool_size)

        pool_size - size of pool
        '''
        self.last_input = None
        self.pool_size = pool_size

    def separate_input_data(self, input_data):
        '''
        Separate the input data into data regions 
        with (pool_size x pool_size) dimensions.

        input_data - the input data
        return the data regions and the location
        '''
        # get the dimensions of the input data
        row_num, col_num, _ = input_data.shape
        for i in range(row_num // self.pool_size):
            for j in range(col_num // self.pool_size):
                i_min = self.pool_size * i
                j_min = self.pool_size * j
                i_max = self.pool_size * (i + 1)
                j_max = self.pool_size * (j + 1)
                input_data_region = input_data[i_min:i_max, j_min:j_max]
                yield input_data_region, i, j

    def forward(self, input_data):
        '''
        Perform the forward pass of the max pooling layer with the filters. 

        input_data - the input data
        return the filtered images.
        '''
        # store the input_data for the back propagation
        self.last_input = input_data

        # initialize the output with the dimensions of the input and pool
        row_num, col_num, filter_num = input_data.shape
        output = np.zeros((row_num // self.pool_size, col_num // self.pool_size, filter_num))

        # perform the max pooling layer
        for input_data_region, i, j in self.separate_input_data(input_data):
            output[i, j] = np.amax(input_data_region, axis=(0, 1))
        return output

    def backprop(self, dL_dout):
        '''
        Perform the back propagation of the max pooling layer. 
        Calculate the gradient and pass the output to the
        convolutional layer

        dL_dout - the loss gradient for this layer's outputs
        return the loss gradient for this layer's inputs
        '''
        # initialize the gradient of the max pooling layer
        dL_din = np.zeros(self.last_input.shape)

        # calculate the gradient
        for input_data_region, i, j in self.separate_input_data(self.last_input):
            row_num, col_num, filter_num = input_data_region.shape
            max_list = np.amax(input_data_region, axis=(0, 1))

            for row in range(row_num):
                for col in range(col_num):
                    for fil in range(filter_num):
                        if input_data_region[row, col, fil] == max_list[fil]:
                            dL_din[i * self.pool_size + row, j * self.pool_size + col, fil] = dL_dout[i, j, fil]

        return dL_din

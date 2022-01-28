import numpy as np


class SoftMax:
    '''
    A standard fully-connected layer with softmax activation
    '''
    def __init__(self, image, filter_num, filter_size, pool_size, node_num):
        '''
        Initialize a standard fully-connected layer with softmax activation

        image - the input image
        filter_num - number of filters
        filter_size - size of filters
        pool_size - size of pool
        node_num - number of nodes
        '''
        self.last_total_list = None
        self.last_input = None
        self.last_input_shape = None

        # calculate the total number of inputs of the fully-connected layer
        row_num, col_num = image.shape
        conv_row_num = row_num - filter_size + 1
        conv_col_num = col_num - filter_size + 1
        pool_row_num = conv_row_num // pool_size
        pool_col_num = conv_col_num // pool_size
        input_num = pool_row_num * pool_col_num * filter_num

        # randomly generate the weights and bias for the first time
        self.weight_list = np.random.randn(input_num, node_num) / input_num
        self.bias_list = np.zeros(node_num)

    def forward(self, input_data):
        '''
        Perform the forward pass of the fully-connected layer with softmax activation. 

        input_data - the input data
        return an array containing the respective probability values.
        '''
        # store the dimension information of the input
        self.last_input_shape = input_data.shape

        # flatten the input data to be processed
        input_data = input_data.flatten()

        # store the input data
        self.last_input = input_data     

        # calculate the output using softmax
        total_list = np.dot(input_data, self.weight_list) + self.bias_list
        self.last_total_list = total_list

        exp_list = np.exp(total_list)
        output = exp_list / np.sum(exp_list, axis=0)
        return output

    def backprop(self, dL_dout, learn_rate):
        '''
        Perform the back propagation of the convolutional
        layer. Calculate the gradient, set the new weights and biases
        and pass the result to the previous layer.

        dL_dout - the loss gratient for this layer's outputs
        learn_rate - the learning rate of the layer
        '''
        for i, grad in enumerate(dL_dout):
            if grad == 0:
                continue

            t_exp = np.exp(self.last_total_list)

            t_exp_sum = np.sum(t_exp)

            dout_dt = -t_exp[i] * t_exp / (t_exp_sum ** 2)
            dout_dt[i] = t_exp[i] * (t_exp_sum - t_exp[i]) / (t_exp_sum ** 2)

            dt_dw = self.last_input
            dt_db = 1
            dt_din = self.weight_list

            dL_dt = grad * dout_dt

            dL_dw = dt_dw[np.newaxis].T @ dL_dt[np.newaxis]
            dL_db = dL_dt * dt_db
            dL_din = dt_din @ dL_dt

            self.weight_list -= learn_rate * dL_dw
            self.bias_list -= learn_rate * dL_db
            return dL_din.reshape(self.last_input_shape)

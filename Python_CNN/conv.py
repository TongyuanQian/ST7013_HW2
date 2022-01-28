import numpy as np


class ConvLayer:
    '''
    This is a convolutional layer that using n x n filters
    '''
    def __init__(self, filter_num, filter_size):
        '''
        Initialize a list of filters with dimensions:
        (filter_num, filter_size, filter_size)

        filter_num - number of filters
        filter_size - size of filters
        '''
        self.last_input = None
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.filter_list = np.random.randn(filter_num, filter_size, filter_size) / (filter_size ** 2)

    def separate_input_image(self, input_image):
        '''
        Separate the input image into image regions 
        with (filter_size x filter_size) dimensions.

        input_image - the input image
        return the image regions and the location
        '''
        # get the dimensions of the input image
        row_num, col_num = input_image.shape
        for i in range(row_num - self.filter_size + 1):
            for j in range(col_num - self.filter_size + 1):
                input_image_region = input_image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield input_image_region, i, j

    def forward(self, input_image):
        '''
        Perform the forward pass of the convolutional layer with the filters. 

        input_image - the input image
        return the filtered images.
        '''
        # store the input_image for the back propagation
        self.last_input = input_image    

        # initialize the output with the dimensions of the input and filters
        row_num, col_num = input_image.shape
        out_row_num = row_num - self.filter_size + 1
        out_col_num = col_num - self.filter_size + 1
        output = np.zeros((out_row_num, out_col_num, self.filter_num))

        # perform the convolution
        for input_image_region, i, j in self.separate_input_image(input_image):
            output[i, j] = np.sum(input_image_region * self.filter_list, axis=(1, 2))
        return output

    def backprop(self, dL_dout, learn_rate):
        '''
        Perform the back propagation of the convolutional
        layer. Calculate the gradient and set the new set of filters.

        dL_dout - the loss gratient for this layer's outputs
        learn_rate - the learning rate of the layer
        '''
        # initialize the gradient of the filters
        dL_dfilter = np.zeros(self.filter_list.shape)

        # calculate the gradient
        for input_image_region, i, j in self.separate_input_image(self.last_input):
            for fil in range(self.filter_num):
                dL_dfilter[fil] += dL_dout[i, j, fil] * input_image_region

        # set the filters with the gradient
        self.filter_list -= learn_rate * dL_dfilter

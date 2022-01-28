import matplotlib.pyplot as plt
import numpy as np

class ReadImage:
    def read_image(self, image_index):
        image_name = str(image_index) + '.jpg'
        origin_image = plt.imread(image_name)
        row_num, col_num, _ = origin_image.shape
        greyscale_image = np.zeros((row_num, col_num))
        for i in range(row_num):
            for j in range(col_num):
                greyscale_image[i, j] = 255 - origin_image[i, j, 0]

        return greyscale_image

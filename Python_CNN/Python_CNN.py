import mnist
import numpy as np
import matplotlib.pyplot as plt
from Python_ReadImage import ReadImage
from conv import ConvLayer
from pool import MaxPool
from softmax import SoftMax

train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

conv = ConvLayer(filter_num=2, filter_size=6)
pool = MaxPool(pool_size=4)
softmax = SoftMax(mnist.test_images()[0], filter_num=2, filter_size=6, pool_size=4, node_num=10)
ri = ReadImage()

def forward(image, label):
    conv_out = conv.forward((image / 255) - 0.5)
    pool_out = pool.forward(conv_out)
    out = softmax.forward(pool_out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


def train(image, label, learn_rate):
    out, loss, acc = forward(image, label)

    grad = np.zeros(10)
    grad[label] = -1 / out[label]

    grad = softmax.backprop(grad, learn_rate)
    grad = pool.backprop(grad)
    conv.backprop(grad, learn_rate)

    return loss, acc

def my_test():
    print('\n--- Testing with my own cases ---')
    num_correct = 0
    loss = 0
    for i in range(10):
        image = ri.read_image(i)
        _, l, acc = forward(image, i)
        loss += l
        num_correct += acc
    print('Average Loss = %.3f | Correct Number = %d' % (loss / 10, num_correct ))

# Train the CNN for 3 epochs
for epoch in range(3):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Train!
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0

        l, acc = train(im, label, 0.005)
        loss += l
        num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)


my_test()
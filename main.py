import net
import mnist_loader

net = net.Net([784, 10, 10])

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


net.SGD(training_data, 3, 250, 6.0, test_data=test_data)

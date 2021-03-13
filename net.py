# Simple implementation of feedforward neural network, 
# stochastic gradient descent, 
# backprop gradients 

## Based on numpy 
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Net(object):

    def __init__(self, sizes):
	
        #sizes represents the neuron distribution, as in [3,4,5,2]
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        
        # LOG of biases and weights for optimization purposes 
        self.biases_log = []
        self.weights_log = []
        
        # One for each neuron 
        bias_matrix_dim = sizes[1:]
        # One for each connection between neurons
        
        weight_matrix_dim = [sizes[:-1], sizes[1:]]
        print("bias matrix dim", bias_matrix_dim)
        print("weight matrix dim",weight_matrix_dim)

        self.biases = self.set_biases(bias_matrix_dim)
        self.weights = self.set_weights(weight_matrix_dim)

    def set_biases(self,bias_matrix_dim):
        return [np.random.randn(y, 1) for y in bias_matrix_dim]
    
    def set_weights(self,weight_matrix_dim):
        return [np.random.randn(y, x) for x, y in zip(weight_matrix_dim[0],weight_matrix_dim[1])]
    

    def feedforward(self, a):
	#Function for evaluation 
	# Vectorized calc. and results if input is a         
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def split_data(self, training_data,mini_batch_size): 
            n = len(training_data)
            return [training_data[k:k+mini_batch_size]for k in range(0, n, mini_batch_size)]


    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
	# eta as learning rate 
	# setup the test enviroment 
        training_data = list(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
	# cycle trough the requested number of epochs 
        for j in range(epochs):
	    # Shuffle the training data, so to achieve random batches
            random.shuffle(training_data)
	    # A bit of code magic for producing the batches 
            mini_batches = self.split_data(training_data, mini_batch_size)
            print("mini batch len : ", len(mini_batches))
	    # Now iterate trough the batches and do the backprop
            for mini_batch in mini_batches:
                #update weights and biases 
                self.update(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
                
            # Following commands for plotting 
            
            #level = 0 # choose which level of the network to plot 
            #self.biases_log.append(self.biases[level])
            #self.weights_log.append(self.weights[level])
         
        
        #self.plot_biases()
        #self.plot_weights()
    
    def plot_biases(self):
        
        fig = plt.figure()
        
        plt.title("Bias animation, made of {0} frames".format(len(self.biases_log)))
        plot =plt.imshow(self.biases_log[0],animated=True)
        fig.colorbar(plot)
        
        def init():
            plot.set_data(self.biases_log[0])
            return [plot]
    
        def update(j):
            plot.set_data(self.biases_log[j])
            return [plot]
        
        animation.FuncAnimation(fig, update, init_func = init, frames=len(self.biases_log), interval = 180, blit=True)
        plt.show()


    def plot_weights(self):
        
        fig = plt.figure()
        
        plt.title("Weights animation, made of {0} frames".format(len(self.weights_log)))
        plot =plt.imshow(self.weights_log[0],animated=True)
        fig.colorbar(plot)
        
        def init():
            plot.set_data(self.weights_log[0])
            return [plot]
    
        def update(j):
            plot.set_data(self.weights_log[j])
            return [plot]
        
        animation.FuncAnimation(fig, update, init_func = init, frames=len(self.weights_log), interval = 500, blit=True)
        plt.show()


    def update(self, mini_batch, eta):

	# Prepare the arrays for the nablas weights and biases
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
	# for each training example, we run the backprop trough the network
        for x, y in mini_batch:
	    # we obtain from the network the two vectors 
	    # containing the partial derivatives with respect to weight and bias
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
	    #sum the value for reconstructing the contribution
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
	# finally compute the updated values of weights and biases 
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        
        

    def backprop(self, x, y):
	# Define the two vectors representing gradient 
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
	# run forward in the network 
        for b, w in zip(self.biases, self.weights):
	    #compute each intermediate value z
            z = np.dot(w, activation)+b
	    #save it for later 
            zs.append(z)
	    #calculate and store each activation value 
            activation = sigmoid(z)
            activations.append(activation)
        # backward run trough the network 
	# calculate the first delta 
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
	# and then all the others running backward 
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
	    # calculate current delta 
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
	    # calculate each value for bias and weight
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
	#return the two vectors 
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # external derivative calculation
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

# Sigmoid function and derivative
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

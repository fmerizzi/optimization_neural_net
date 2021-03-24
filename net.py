import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mnist_loader
import time 
import copy


def set_biases(bias_matrix_dim,rnd):
	return [rnd.randn(y, 1) for y in bias_matrix_dim]
    
def set_weights(weight_matrix_dim,rnd):
	return [rnd.randn(y, x) for x, y in zip(weight_matrix_dim[0],weight_matrix_dim[1])]

def split_data(training_data,mini_batch_size): 
            n = len(training_data)
            return [training_data[k:k+mini_batch_size]for k in range(0, n, mini_batch_size)]
        
        
def plot_biases(biases_log):
    
        fig, ax = plt.subplots()
        ax.set_xticklabels([0])
        # Calclulate change 
        change = biases_log[-1] - biases_log[0] 
        change = np.mean(change)
                
        plt.title("Bias animation, made of {0} frames. Mean change: {1} ".format(len(biases_log),round(change,4)), loc="right",pad=4.6)
        plot =plt.imshow(biases_log[0],animated=True)
        fig.colorbar(plot)
        
        def init():
            plot.set_data(biases_log[0])
            return [plot]
    
        def update(j):
            plot.set_data(biases_log[j])
            return [plot]
        
        animation.FuncAnimation(fig, update, init_func = init, frames=len(biases_log), interval = 180, blit=True)
        plt.show()


def plot_weights(weights_log):
        
        fig, ax = plt.subplots()
        ax.set_xticklabels([0])
        # Calclulate change 
        change = weights_log[-1] - weights_log[0] 
        change = np.mean(change)
                
        plt.title("weights animation, made of {0} frames. Mean change: {1} ".format(len(weights_log),round(change,4)), loc="right",pad=4.6)
        plot =plt.imshow(weights_log[0],animated=True)
        fig.colorbar(plot)
        
        def init():
            plot.set_data(weights_log[0])
            return [plot]
    
        def update(j):
            plot.set_data(weights_log[j])
            return [plot]
        
        animation.FuncAnimation(fig, update, init_func = init, frames=len(weights_log), interval = 180, blit=True)
        plt.show()


def train(training_data, epochs, mini_batch_size, eta,test_data,weights,biases):
    
        start = time.time()
        train_evaluation = []
        timestamps =  []
        #Setup of logs for visualization purposes 
        biases_log = []
        weights_log = []
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
            mini_batches = split_data(training_data, mini_batch_size)
            print("mini batch len : ", len(mini_batches))
	    # Now iterate trough the batches and do the backprop
            for mini_batch in mini_batches:
            		#update weights and biases 
            		# Prepare the arrays for the nablas weights and biases
            		nabla_b = [np.zeros_like(b) for b in biases]
            		nabla_w = [np.zeros_like(w) for w in weights]
            		
            		# for each training example, we run the backprop trough the network
            		for x, y in mini_batch:
            		    # we obtain from the network the two vectors 
            		    # containing the partial derivatives with respect to weight and bias
            		    delta_nabla_b, delta_nabla_w = backprop(x, y,biases,weights)
            		    #sum the value for reconstructing the contribution
            		    nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            		    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            		# finally compute the updated values of weights and biases 
            	    
            		weights = [w-(eta/len(mini_batch))*nw
            		                for w, nw in zip(weights, nabla_w)]
            		biases = [b-(eta/len(mini_batch))*nb
            		               for b, nb in zip(biases, nabla_b)]
        # For testing is simple classification check 
            timestamps.append(time.time() - start)
            if test_data:
                v = evaluate(test_data,biases,weights)
                print("{0} --> correct classifications: ({1} / {2}) ".format(
                    j, v, n_test))
                train_evaluation.append(v)
            else:
                print("Epoch {0} complete".format(j))
                
            # Following commands for plotting 
            
            level = 1 # choose which level of the network to plot 
            biases_log.append(biases[level])
            weights_log.append(weights[level])
        
        plot_biases(biases_log)
        plot_weights(weights_log)
        return train_evaluation, timestamps


def backprop(x, y,biases,weights):
	# Define the two vectors representing gradient 
        nabla_b = [np.zeros_like(b) for b in biases]
        nabla_w = [np.zeros_like(w) for w in weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        #FORWARD RUN 
        # a cycle for 
        for b, w in zip(biases, weights):
        # We compute everything in vector form 
    
	    #compute each intermediate value z
            z = np.dot(w, activation)+b
	    #save it for later 
            zs.append(z)
	    #calculate and store each activation value 
            activation = sigmoid(z)
            activations.append(activation)
        # backward run trough the network 
	# calculate the first delta 
        delta = cost_derivative(activations[-1], y) * sigmoid_deriv(zs[-1])
        nabla_b[-1] = delta
        #BACKWARD RUN 
	# and then all the others running backward 
        for l in range(-2, -len(sizes),-1):
            z = zs[l]
            sp = sigmoid_deriv(z)
	    # calculate current delta 
            delta = np.dot(weights[l+1].transpose(), delta) * sp
	    # calculate each value for bias and weight, as in (BP3/4)
            nabla_b[l] = delta
            nabla_w[l] = np.dot(delta, activations[l-1].transpose())
	#return the two vectors 
        return (nabla_b, nabla_w)

def cost_derivative(output_activations, y):
        return (output_activations-y)

def evaluate(test_data,biases,weights):
        test_results = [(np.argmax(activation(x,biases,weights)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def activation(a,biases,weights):
	#Function for evaluation 
	# Vectorized calc. and results if input is a         
        for b, w in zip(biases,weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))


###############################################################################################################

#Set network architecture
sizes = [784,30,10]
num_layers = len(sizes)

#prepare the bias/weights structures 
bias_matrix_dim = sizes[1:]
weight_matrix_dim = [sizes[:-1], sizes[1:]]

print("bias matrix dim", bias_matrix_dim)
print("weight matrix dim",weight_matrix_dim)

rnd = np.random.RandomState(777)

biases = set_biases(bias_matrix_dim,rnd)
weights = set_weights(weight_matrix_dim,rnd)

train_evaluation = []
timestamps = []
train_evaluation2 = []
timestamps2 = []


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

print(type(training_data),training_data)
print(type(validation_data),validation_data)
print(type(test_data),test_data)



train_evaluation,timestamps = train(copy.deepcopy(training_data), 5, 500, 1,
                                    copy.deepcopy(test_data),copy.deepcopy(weights),copy.deepcopy(biases))
train_evaluation2,timestamps2 = train(copy.deepcopy(training_data),5, 500, 1, 
                                    copy.deepcopy(test_data),weights,biases)


fig2, ax = plt.subplots()
ax.plot(timestamps,train_evaluation,color="red",label="net 1")
ax.plot(timestamps2,train_evaluation2,color="blue",label="net 2")
ax.set(xlabel='time (s)', ylabel='score',
       title='train evaluation')
ax.grid()
plt.show()

 
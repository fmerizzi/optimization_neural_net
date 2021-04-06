import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import loader
import time 
import copy


def set_biases(bias_matrix_dim,rnd):
	return [rnd.randn(y, 1) for y in bias_matrix_dim]
    
def set_weights(weight_matrix_dim,rnd):
	return [rnd.randn(y, x) for x, y in zip(weight_matrix_dim[0],weight_matrix_dim[1])]

def set_weights_unsaturated(weight_matrix_dim,rnd):
    # by dividing the weight by the squared number of neurons we avoid over saturation
    # squish the gaussian 
	return [rnd.randn(y, x)/np.sqrt(x)  for x, y in zip(weight_matrix_dim[0],weight_matrix_dim[1])]

def split_data(training_data,mini_batch_size): 
            n = len(training_data)
            return [training_data[k:k+mini_batch_size]for k in range(0, n, mini_batch_size)]
        
        
def plot_biases(biases_log):
    
        fig, ax = plt.subplots(figsize=[7,5])
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
        
        save = animation.FuncAnimation(fig, update, init_func = init, frames=len(biases_log), interval = 180, blit=True)
        plt.show()
        # save animation? 
        save.save('bias.gif', writer='imagemagick', fps=15)


def plot_weights(weights_log):
        
        #fig, ax = plt.subplots(figsize=[7,5])
        fig, ax = plt.subplots(figsize=[14,10])
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
        
        save = animation.FuncAnimation(fig, update, init_func = init, frames=len(weights_log), interval = 180, blit=True)
        plt.show()
        # save animation?
        save.save('weight.gif', writer='imagemagick', fps=15)


def train(training_data, epochs, mini_batch_size, eta,test_data,weights,biases,rnd,plot=True,L2=True,cross_entropy=False):
        
        # Start time, timestamps for evaluation purposes 
        start = time.time()
        train_evaluation = []
        timestamps =  []
        #Setup of logs for visualization purposes 
        biases_log = []
        weights_log = []

        n_test = len(test_data)
	    # cycle trough the requested number of epochs 
        for i in range(epochs):
	         # Shuffle the training data, so to achieve random batches
            rnd.shuffle(training_data)
	        # A bit of code magic for producing the batches 
            mini_batches = split_data(training_data, mini_batch_size)
            print("mini batch len : ", len(mini_batches))
            
            # SGD
	        # Now iterate trough the batches and do the backprop
            for mini_batch in mini_batches:
            		# update weights and biases 
            		# Prepare the arrays for the nablas weights and biases
                    nabla_b = [np.zeros_like(b) for b in biases]
                    nabla_w = [np.zeros_like(w) for w in weights]

            		# for each training example, we run the backprop trough the network
                    for x, y in mini_batch:
            		    # we obtain from the network the two vectors 
            		    # containing the partial derivatives with respect to weight and bias
                        deriv_bias, deriv_weight = backprop(x, y,biases,weights,cross_entropy)
            		    #sum the values
                        for l in range(len(nabla_b)):
                            nabla_b[l] = nabla_b[l] + deriv_bias[l]
                        for l in range(len(nabla_w)):
                            nabla_w[l] = nabla_w[l] + deriv_weight[l]
                        
            		# finally compute the updated values of weights and biases         
                    # Optional L2 regularization 
                    if(L2==True):
                        weights = [(1-eta*(3/len(training_data)))*w-(eta/len(mini_batch))*nw
            		                for w, nw in zip(weights, nabla_w)]
                    else:
                        weights = [w-(eta/len(mini_batch))*nw
            		                for w, nw in zip(weights, nabla_w)]
                        
                    biases = [b-(eta/len(mini_batch))*nb
            		               for b, nb in zip(biases, nabla_b)]
            
            # For testing is simple classification check 
            timestamps.append(time.time() - start)
            
            #Get current net performance 
            v = evaluate(test_data,biases,weights)
            print("{0} --> correct classifications: ({1} / {2}) ".format(
                i, v, n_test))
            # update the train evaluation list 
            train_evaluation.append(v)

                
            # Following commands for plotting 
            if(plot==True):
                level = 1 # choose which level of the network to plot 
                biases_log.append(biases[level])
                weights_log.append(weights[level])
        
        if(plot==True):
            plot_biases(biases_log)
            plot_weights(weights_log)
        return train_evaluation, timestamps


def backprop(x, y,biases,weights,cross_entropy):
    
        net_input = x
        a = [] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        
	    # Define the two empty vectors representing the cost gradient 
        deriv_bias = [np.zeros_like(b) for b in biases]
        deriv_weight = [np.zeros_like(w) for w in weights]
        
        a.append(net_input)
       
        for b, w in zip(biases, weights):
        # We compute everything in vector form 
    
	    #compute each intermediate value z
            z = np.dot(w, net_input)+b
	    #save it for later 
            zs.append(z)
	    #calculate and store each activation value 
            net_input = sigmoid(z)
            a.append(net_input)
        # backward run trough the network 
	    # calculate the first delta 
        if(cross_entropy == True):
            delta = crossCost_derivarive(a[-1], y,zs[-1]) 
        else:
            delta = quadcost_derivative(a[-1], y,zs[-1]) 
            
        deriv_bias[-1] = delta
        #BACKWARD RUN 
	# and then all the others running backward 
        for l in range(-2, -len(sizes),-1):
            z = zs[l]
            sigma_prime = sigmoid_deriv(z)
	    # calculate current delta 
        # Hadamard product *
            delta = np.dot(weights[l+1].transpose(), delta) * sigma_prime
	    # calculate each value for bias and weight, as in (BP3/4)
            deriv_bias[l] = delta
            deriv_weight[l] = np.dot(delta, a[l-1].transpose())
	#return the two vectors 
        return (deriv_bias, deriv_weight)

def quadcost_derivative(output_activations, y,z):
        return ((output_activations-y) * sigmoid_deriv(z))
def crossCost_derivarive(output_activations, y,z):
        #Simply remove the activation term
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
    try:
        return 1.0/(1.0+np.exp(-z))
    except:
        print(z)
    
def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))


###############################################################################################################

#Set network architecture
sizes = [784,28,28,10]
num_layers = len(sizes)

#prepare the bias/weights structures 
bias_matrix_dim = sizes[1:]
weight_matrix_dim = [sizes[:-1], sizes[1:]]

print("bias matrix dim", bias_matrix_dim)
print("weight matrix dim",weight_matrix_dim)

rnd = np.random.RandomState(311)

biases = set_biases(bias_matrix_dim,copy.deepcopy(rnd))
# Using the unsaturated version dramatically improves the first stages of learning
weights = set_weights_unsaturated(weight_matrix_dim,copy.deepcopy(rnd))

train_evaluation = []
timestamps = []
train_evaluation2 = []
timestamps2 = []

# Load the data, as zip iterables 
training_data, test_data = loader.load_data()

# Hyperparameter set 
epochs = [1,7]
mini_batch_size = [100,200]
eta = [1.5,1.5]


# Set error so to be reactive to overflow 
np.seterr(all='print')

# Call the two train methods 
train_evaluation,timestamps = train(copy.deepcopy(training_data), epochs[0], mini_batch_size[0], eta[0],
                                    copy.deepcopy(test_data),copy.deepcopy(weights),copy.deepcopy(biases),copy.deepcopy(rnd),
                                    plot=False,
                                    L2=True,
                                    cross_entropy=True)
train_evaluation2,timestamps2 = train(copy.deepcopy(training_data),epochs[1], mini_batch_size[1], eta[1], 
                                    copy.deepcopy(test_data),copy.deepcopy(weights),copy.deepcopy(biases),copy.deepcopy(rnd),
                                    plot=True,
                                    L2=False,
                                    cross_entropy=True)


# Plot a train evaluation
fig2, ax = plt.subplots()
ax.plot(timestamps,train_evaluation,color="red",label="1- batch: {0}, eta: {1}, epochs: {2}".format(mini_batch_size[0],eta[0],epochs[0]))
ax.plot(timestamps2,train_evaluation2,color="blue",label="2- batch: {0}, eta: {1}, epochs: {2}".format(mini_batch_size[1],eta[1],epochs[1]))
ax.set(xlabel='time (s)', ylabel='score',
       title='train evaluation')
ax.grid()
ax.legend(shadow=True, fontsize="large")

plt.show()

  

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import loader
import time 
import copy

def set_biases(bias_matrix_dim,rnd):
    tmp = []
    for i in bias_matrix_dim:
        tmp.append(rnd.randn(i, 1))
    return tmp
    
def set_weights(weight_matrix_dim,rnd):
    tmp = []
    for i in weight_matrix_dim:
        tmp.append(rnd.randn(i[1], i[0]))
    return tmp

def set_weights_unsaturated(weight_matrix_dim,rnd):
    tmp = []
    for i in weight_matrix_dim:
        tmp.append(rnd.randn(i[1], i[0])/np.sqrt(i[0]))
    return tmp
    
def split_data(training_data,mini_batch_size): 
    tmp = []
    for i in range(0,len(training_data),mini_batch_size):
        tmp.append(training_data[i:i+mini_batch_size])
    return tmp

def prepare_weightMatrix_dim(sizes):
    tmp = []
    for i in range(len(sizes)-1):
        tmp.append([sizes[i],sizes[i+1]])
    return tmp
        
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
	        # split the data in batches
            mini_batches = split_data(training_data, mini_batch_size)
            print("batches number : ", len(mini_batches))
            
            # SGD
	        # Update paramenters for each mini batch
            for mini_batch in mini_batches:
            		
            		# Prepare the lists for the partial derivatives of weights and biases
                    partial_deriv_biases = [np.zeros_like(b) for b in biases]
                    partial_deriv_weights = [np.zeros_like(w) for w in weights]

            		# for each training example, we run the backprop trough the network
                    for x, y in mini_batch:
            		    # we obtain from the network the two vectors 
            		    # containing the partial derivatives with respect to weight and bias
                        deriv_bias, deriv_weight = backprop(x, y,biases,weights,cross_entropy)
            		    #sum the values
                        for l in range(len(partial_deriv_biases)):
                            partial_deriv_biases[l] = partial_deriv_biases[l] + deriv_bias[l]
                        for l in range(len(partial_deriv_weights)):
                            partial_deriv_weights[l] = partial_deriv_weights[l] + deriv_weight[l]
                        
            		# finally compute the updated values of weights and biases         
                    # Optional L2 regularization 
                    if(L2==True):
                        for l in range(len(weights)):
                            weights[l] = (1-(4/len(training_data))*weights[l]) - (eta/len(mini_batch))*partial_deriv_weights[l]
                    else:
                        for l in range(len(weights)):
                            weights[l] = weights[l] - (eta/len(mini_batch))*partial_deriv_weights[l]

                    for l in range(len(biases)):
                            biases[l] = biases[l] - (eta/len(mini_batch))*partial_deriv_biases[l]

            
            # time evaluation
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


def backprop(net_input, y,biases,weights,cross_entropy):
    
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
        # TODO 
        # In a better way
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

# PARAMETER SETTING 
#Set network architecture
sizes = [784,10,10]
#Set if unsaturated weights
unsaturated_weights = True
# Hyperparameter set 
epochs = [5,5]
mini_batches = [100,200]
eta = [1.5,3]

###############################################################################################################
num_layers = len(sizes)
# Prepare biases for each level except the first 
bias_matrix_dim = sizes[1:]
#weight_matrix_dim = [sizes[:-1], sizes[1:]]
weight_matrix_dim = prepare_weightMatrix_dim(sizes)


print("bias matrix dim", bias_matrix_dim)
print("weight matrix dim",weight_matrix_dim)

rnd = np.random.RandomState(1)

biases = set_biases(bias_matrix_dim,copy.deepcopy(rnd))
# Using the unsaturated version dramatically improves the first stages of learning
if(unsaturated_weights==True):
    weights = set_weights_unsaturated(weight_matrix_dim,copy.deepcopy(rnd))
else:
    weights = set_weights(weight_matrix_dim,copy.deepcopy(rnd))


train_evaluation = []
timestamps = []
train_evaluation2 = []
timestamps2 = []

# Load the data, as zip iterables 
training_data, test_data = loader.load_data()


# Set error so to be reactive to overflow 
np.seterr(all='print')

# Call the two train methods 
train_evaluation,timestamps = train(copy.deepcopy(training_data), epochs[0], mini_batches[0], eta[0],
                                    copy.deepcopy(test_data),copy.deepcopy(weights),copy.deepcopy(biases),copy.deepcopy(rnd),
                                    plot=True,
                                    L2=False,
                                    cross_entropy=False)
train_evaluation2,timestamps2 = train(copy.deepcopy(training_data),epochs[1], mini_batches[1], eta[1], 
                                    copy.deepcopy(test_data),copy.deepcopy(weights),copy.deepcopy(biases),copy.deepcopy(rnd),
                                    plot=True,
                                    L2=False,
                                    cross_entropy=True)


# Plot a train evaluation
fig2, ax = plt.subplots()
ax.plot(timestamps,train_evaluation,color="red",label="1- batch: {0}, eta: {1}, epochs: {2}".format(mini_batches[0],eta[0],epochs[0]))
ax.plot(timestamps2,train_evaluation2,color="blue",label="2- batch: {0}, eta: {1}, epochs: {2}".format(mini_batches[1],eta[1],epochs[1]))
ax.set(xlabel='time (s)', ylabel='score',
       title='train evaluation')
ax.grid()
ax.legend(shadow=True, fontsize="large")

plt.show()

  

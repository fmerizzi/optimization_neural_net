
# Neural network optimizaton project AI_unibo_2021

Simple neural network implementation, based on KMNIST (japanese ideograms) classification.
net.py contains all the necessary functions. 
loader.py load the kmnist dataset and applies basic data transformations. 

## Functionalities 

### Vanilla network
1)  backpropagation algorithm with SGD 
2)  basic sigmoid activation function
3)  quadratic cost function 
4)  random gaussian weight inizialization 
### Improovements 
1) weight inizialization
2) L2 regularization
3) cross entropy cost function
4) learning rate scheduler?
5) momentum based SGD? 
### Other features 
6)  animation showing the change in bias and weights matrix
7)  basic train evaluation tools: time, accuracy
8)  basic hyperparamenter tuning with graph equalizing time, random initializaions
9)  shows vanishing gradient problem 

## Backprop algorithm visualization 

![backprop](https://user-images.githubusercontent.com/32902835/110661686-1c673b80-81c5-11eb-8117-ff8f0a7c6c7d.png)

## Examples of weight and bias animation
The program saves mid-training weights and biases for visualization purposes. The animation shows how the network change its parameters with gradient descent. It is possible to show the vanishing gradient problem by inspecting the first layer. 

![weight](https://user-images.githubusercontent.com/32902835/112352990-91f60000-8ccb-11eb-814a-a8919fad04d0.gif)

![bias](https://user-images.githubusercontent.com/32902835/112353099-b2be5580-8ccb-11eb-8cbd-0e8440df05ae.gif)

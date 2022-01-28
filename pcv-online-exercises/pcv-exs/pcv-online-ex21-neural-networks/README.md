# Neural Networks

## 5 Minutes
- They are used in AI, machine learning
- Take input signal, do some computation on that input signal and generate output (this mean they are functions)
- How do those function look like and how can we specify them ? 
- Neural network contains neurons: they are small computational units. These unit are connected as a network in some way
- Chaining multiple neurons allows for more complex functions.
- Neural network is a concatenation of such simple functions 
- To create a neural network, we need to specify the network with all its neurons and parameters ... 
- Finding good parameters can be very hard -> We learn all these parameters.
- We learn parameters by showing it example images of cats, dogs, ....
- Set the parameters so that the network can classify the given examples correctly.
- Gradient Descent, Stochastic Gradient Descent ,... use to minimize the mismatch between the network output and the true
values. 
- Backpropagation is a technique to compute the gradient of the loss w.r.t the parameters of the network. 
- There exists a large zoo of neural networks (the topology of the network). In usual, we fix the topology and then learn
the parameters from this topology.
- Neural networks are often used when interpreting sensor data.

## Part 1: The Basics of Neural Networks 

Image Classification: single label for an image

Semantic Segmentation: single label for each pixel

* Machine learning technique 
* Often used for classification, semantic segmentation and related tasks 
* Fist ideas discussed in the 1950/1960s
* Theory work on NNs in the 1990s
* Increase in attention from 2000
* Deep learning took off around 2010 
* CNNs for image tasks from 2012

"Neural Network" include Neuron and Network

Neuron: fundamental unit of the brain
Network: Connected elements 

Neural Network include several neurons that are somehow connected. They get information from sensor and they can process 
this information by doing basic computing by forwarding certain signals to other neurons until we reach a certain output 
neuron

### Artificial Neurons
Artificial neurons are the fundamental units of artificial neural networks that: 
* Receive inputs 
* Transform information 
* Create an output 

### Neuron
* Receive inputs/ activations from sensors or other neurons
* Combine/transform information 
* Create an output/activation 

### Neurons as Functions
We can see a neuron as a function:
* Input given by x in R^N (N-D input)
* Transformation of the input data can be described bt a function f
* Output f(x) = y in R (1-D output)

### Neural Network
* Neural Network is a network/graph of neurons
* Nodes are neurons
* Edges represent input-output connection of the data flow

### Neural Network as a Function
* The whole network is again a function (a more flexible functions)
* Multi-layer perceptron or MLP is often seen as the "vanilla" neural network. In MLP, data flows always from one side 
to the other side. Data forwarded in one direction so there are no loop. 
* Input layer takes (sensor) data 
* Output layer provides the function result (information or command)
* Hidden layers do some computations

### Different Types of NNs 
* Perceptron 
* MLP-Multilayer perceptron
* Autoencoder
* CNN - Convolutional NN
* RNN - Recurrent NN 
* LSTM - Long/short term memory NN 
* GANs - Generative adversarial network 
* Graph NN
* Transformer
* ... 

### Multilayer Perceptron Seen as a Function








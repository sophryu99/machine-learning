## Neural Network



Neural Network: computational model that is inspired by the way biological neural networks in the human brain process information

> The basic unit of computation in a neural network is the neuron , often called a node or unit. It receives input from some other nodes, or from an external source and computes an output. Each input has an associated
> weight (w), which is assigned on the basis of its relative importance to other inputs. The node applies a function to the weighted sum of its inputs.



### Neural Network Architecture

- **Input Nodes (input layer)**: No compuation is done here within this layer, they just pass the information to the next layer.
- **Hidden Nodes (hidden layer)**: Intermediate processing or computation is done. 
- **Output Nodes (output layer):** Here we finally use an activation function that maps to the desired output format (e.g. softmax for classification).
- **Connections and weights:** The *network* consists of connections, each connection transferring the output of a neuron i to the input of a neuron *j*. In this sense *i* is the predecessor of *j* and *j* is the successor of *i*, Each connection is assigned a weight *Wij.*
- **Activation function:** the **activation function** of a node defines the output of that node given an input or set of inputs. A standard computer chip circuit can be seen as a digital network of activation functions that can be “ON” (1) or “OFF” (0), depending on input. This is similar to the behavior of the linear perceptron in neural networks. However, it is the *nonlinear* activation function that allows such networks to compute nontrivial problems using only a small number of nodes. In artificial neural networks this function is also called the transfer function.
- **Learning rule:** The *learning rule* is a rule or an algorithm which modifies the parameters of the neural network, in order for a given input to the network to produce a favored output. This *learning* process typically amounts to modifying the weights and thresholds.

<br />



### ANN vs CNN vs RNN

#### Artificial Neural Networks (ANN)

Artificial Neural Network, or ANN, is a group of multiple perceptrons/ neurons at each layer. ANN is also known as a **Feed-Forward Neural network** because inputs are processed only in the forward direction:

<img width="400" alt="Screen Shot 2021-01-17 at 3 47 39 PM" src="https://user-images.githubusercontent.com/46921003/104833351-5742a680-58db-11eb-9ff0-67c3be752410.png">

ANN can be used to solve problems related to:

- Tabular data
- Image data
- Text data



**Advantages of ANN**

> An activation function is a powerhouse of ANN

Artificial Neural Network is capable of learning any nonlinear function. Hence, these networks are popularly known as **Universal Function Approximators**. ANNs have the capacity to learn weights that map any input to the output.

<br />



### Convolution Neural Networks (CNN)



<br />



### Recurrent Neural Networks (RNN)

> A looping constraint on the hidden layer of ANN turns to RNN








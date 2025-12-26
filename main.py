import sys
import numpy as np
import matplotlib

class Neuron():
    def __init__(self, inputs,weights,bias):
        """
        Below is a manual implementation of the dot product function,
        which can be performed with np.dot(a,b).
        
        output = 0
        for i in range(len(inputs)):
            output+=inputs[i]*weights[i]
        output += bias
        """
        
        output=np.dot(weights,inputs)+bias
        self.output = output

# Constants   
inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
weights = [weights1,weights2,weights3]

bias1 = 2
bias2 = 3
bias3 = 0.5    
biases = [bias1,bias2,bias3]

""" 
All neurons in a layer have the same inputs (the output of the previous layer),
but output different values because of their unique weights and biases.
"""

layer_outputs = []
neurons = []
for i in range(3):
    neurons.append(Neuron(inputs,weights[i],biases[i]))
    layer_outputs.append(neurons[-1].output)

output = np.dot(weights,inputs)+biases
print(f"Direct np.dot(): {output}")

print(f"Neuron output: {layer_outputs}")
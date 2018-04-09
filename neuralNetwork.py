import random
import numpy
import scipy.special

#Tate Conlon - 10067255
#Running this script requires installing numpy and scipy on your machine.
#Python version 3.6
#Script was written by reading "Make Your Own Neural Network" by Tariq Rashid

#This class defines a neural network with 1 hidden layer, who's activation
#function is a sigmoid and uses momentum
class neuralNetwork:

	def __init__(self, inputnodes, hiddennodes, outputnodes):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		#scipy.special.expit = sigmoid function
		self.activation_function = lambda x: scipy.special.expit(x)

		#setting weight matricies
		#w_i_j where link is from node i to node j
		#wih is weights from input to hidden layer
		#who is weights from hidden to output
		self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
		#previous used for momentum
		self.prev_delta_wih = numpy.zeros((self.hnodes, self.inodes))
		self.prev_delta_who = numpy.zeros(((self.onodes, self.hnodes)))

	#takes input to neural network and returns output
	def evaluate(self, inputs_list):
		#Takes input list, puts it a 2D array (size 1xn) and transposes it 
		#to (nx1) for later matrix multiplication
		inputs = numpy.array(inputs_list, ndmin=2).T
		#TODO: Generalize for arbitrary num layers
		#hidden nodes inputs (sum before activation function)
		hidden_inputs = numpy.dot(self.wih, inputs)
		#hidden layer's output (applying the activation fuction to the summed inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		return final_outputs

	def clone(self):
		temp_child = neuralNetwork(self.inodes, self.hnodes, self.onodes)
		for i in range(len(self.wih)):
			for j in range(len(self.wih[0])):
				temp_child.wih[i][j] = self.wih[i][j]
		for i in range(len(self.who)):
			for j in range(len(self.who[0])):
				temp_child.who[i][j] = self.who[i][j]
		return temp_child

	def __str__(self):
		return "'" + str(self.wih) + "|" + str(self.who) + "'"

SWAP_CHANCE = 0.5

def breed(net1, net2, num_children=2):
	children = []
	for i in range(num_children):
		temp_child = neuralNetwork(net1.inodes, net1.hnodes, net1.onodes)
		for i in range(len(net1.wih)):
			for j in range(len(net1.wih[0])):
				if random.random() < SWAP_CHANCE:
					temp_child.wih[i][j] = net2.wih[i][j]
				else:
					temp_child.wih[i][j] = net1.wih[i][j]

			for i in range(len(net1.who)):
				if random.random() < SWAP_CHANCE:
					temp_child.who[i][j] = net2.who[i][j]
				else:
					temp_child.who[i][j] = net1.who[i][j]

		children.append(temp_child)

	return children

MUTATE_CHANCE = 0.85
MUTATE_PARAM = 0.5
def mutate(net):
	for i in range(len(net.wih)):
		if random.random() < MUTATE_CHANCE:
			net.wih[i] += random.uniform(MUTATE_PARAM*-1, MUTATE_PARAM)
	for i in range(len(net.who)):
		if random.random() < MUTATE_CHANCE:
			net.who[i] += random.uniform(MUTATE_PARAM*-1, MUTATE_PARAM)
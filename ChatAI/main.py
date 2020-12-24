#Keith Flagg
#A basic python chat AI preprocessor 
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy 
import tflearn
import tensorflow as tf
import random
import json
import pickle

with open("intents.json") as file:
	data = json.load(file)

try:
	#save data as bytes
	with open("data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)

except:
	#blank list
	words = []
	labels = []
	docs_x = []
	docs_y = []

	#loops through each intent
	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			#stemming brings each pattern to its root word
			#tokenize words
			twords = nltk.word_tokenize(pattern)
			words.extend(twords)
			docs_x.append(twords)
			docs_y.append(intent["tag"])

			if intent["tag"] not in labels:
				labels.append(intent["tag"])
	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	#set removes duplicates and list converts to a list
	words = sorted(list(set(words)))

	labels = sorted(labels)

	#match word appearance with lists (bag of words)
	training = []
	output = []

	#0 for all classes if tag exists put a 1
	output_empty = [0 for _ in range(len(labels))]

	for x, doc in enumerate(docs_x):
		bag = []

		twords = [stemmer.stem(w) for w in doc]

		for w in words:
			if w in twords:
				bag.append(1)
			else:
				bag.append(0)

		output_row = output_empty[:]
		output_row[labels.index(docs_y[x])] = 1

		training.append(bag)
		output.append(output_row)

	training = numpy.array(training)
	output = numpy.array(output)

	with open("data.pickle", "wb") as f:
		pickle.dump((words, labels, training, output), f)

#Begin AI 
tf.compat.v1.reset_default_graph()

#input layer
net = tflearn.input_data(shape = [None, len(training[0])])
#8 neurons for first hidden layer
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
#produces a probability for each neuron
net = tflearn.fully_connected(net,len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
	#load pre-existing model
	#model.load("model.tflearn")
	#retrains the AI
	keith.py
except:
	#number of training cycles
	model.fit(training, output, n_epoch=2000, batch_size = 8, show_metric = True)
	model.save("model.tflearn")

def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for sent in s_words:
		for i, w in enumerate(words):
			if w == sent:
				bag[i] = 1

	return numpy.array(bag)

def chat():
	print("The AI is ready! (Type to Start chatting) or (\"Quit\" to end the session) ")
	while True:
		inpt = input("You: ")
		if inpt.lower() == "quit":
			break

		results = model.predict([bag_of_words(inpt, words)])
	
		results_index = numpy.argmax(results)	
		
		#finds tag related to user input
		tag = labels[results_index]

		#model[row, column]
		if results[0, results_index] > 0.7:

			for tg in data["intents"]:
				if tg['tag'] == tag:
					responses = tg['responses']

			print("Keithbot: " + random.choice(responses))

		else: 
			print("I couldn't understand what you typed. Please try again.")
			
chat()
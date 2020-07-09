import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from numpy import loadtxt, savetxt

xTrain = loadtxt('train.csv')
yTrain = loadtxt('train_out.csv')
xTest  = loadtxt('test.csv')
yTest  = loadtxt('test_out.csv')

print(xTrain)
print(yTrain)

# define neurons per layer

input_layer  = 4
hidden_layer = 32
output_layer = 4

# build the network
model = Sequential()
model.add(Dense(hidden_layer, activation = 'relu', input_shape = (input_layer,))) #Dense= fully connected
model.add(Dense(output_layer, activation = 'softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 0.001), metrics = ['accuracy'])

# train the network
# batch_size and epochs can be experimented with
# larger batch-size results in better gradient prediciton, lower batch-size results in lesser gradient prediciton and therefore more randomness,
# which can help to get out of local minima
model.fit(xTrain, yTrain, batch_size = 500, epochs = 500, verbose = 2, validation_data = (xTest, yTest))

#extract model parameters
results = model.get_weights()
weights_hidden = results[0]
biases_hidden  = results[1]
weights_output = results[2]
biases_output  = results[3]

#save them in csv files
savetxt('Data/weights_hidden.csv', weights_hidden, fmt = '%10.8f', delimiter = ' ')
savetxt('Data/biases_hidden.csv', biases_hidden, fmt   = '%10.8f', delimiter = ' ', newline = ' ')
savetxt('Data/weights_output.csv', weights_output, fmt = '%10.8f', delimiter = ' ')
savetxt('Data/biases_output.csv', biases_output, fmt   = '%10.8f', delimiter = ' ', newline = ' ')
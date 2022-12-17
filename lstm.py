import numpy as np
import torch
import torch.nn as nn

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        # initialize weights
        self.Wf = torch.randn(hidden_size, input_size+hidden_size)
        self.bf = torch.zeros((hidden_size, 1))
        self.Wi = torch.randn(hidden_size, input_size+hidden_size)
        self.bi = torch.zeros((hidden_size, 1))
        self.Wo = torch.randn(hidden_size, input_size+hidden_size)
        self.bo = torch.zeros((hidden_size, 1))
        self.Wc = torch.randn(hidden_size, input_size+hidden_size)
        self.bc = torch.zeros((hidden_size, 1))
        self.Wy = torch.randn(output_size, hidden_size)
        self.by = torch.zeros((output_size, 1))

    def parameters(self):
    # return an iterator over the model's learnable parameters
        return (self.Wf, self.bf, self.Wi, self.bi, self.Wo, self.bo, self.Wc, self.bc, self.Wy, self.by)
    
    def forward(self, X, h_prev, c_prev):
        # concatenate input and previous hidden state
        X_with_h_prev = torch.cat((X, h_prev), dim=1)
        
        # compute forget gate
        f = sigmoid(np.dot(self.Wf, X_with_h_prev) + self.bf)
        
        # compute input gate
        i = sigmoid(np.dot(self.Wi, X_with_h_prev) + self.bi)
        
        # compute candidate cell state
        c_candidate = np.tanh(np.dot(self.Wc, X_with_h_prev) + self.bc)
        
        # compute cell state
        c = f * c_prev + i * c_candidate
        
        # compute output gate
        o = sigmoid(np.dot(self.Wo, X_with_h_prev) + self.bo)
        
        # compute hidden state
        h = o * np.tanh(c)
        
        # compute output
        y = torch.matmul(self.Wy, h.t()) + self.by
        
        return h, c, y

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


#USE LSTM CLASS

from sklearn.model_selection import train_test_split

# define the training data
data = [("this is a text", "R"), ("another text", "D"), ("some more text", "I")]

# preprocess the data
vocab = set()
for text, label in data:
    for word in text.split():
        vocab.add(word)
vocab = {word: i for i, word in enumerate(vocab)}

X = []
y = []
for text, label in data:
    x = []
    for word in text.split():
        x.append(vocab[word])
    X.append(x)
    if label == "R":
        y.append(0)
    elif label == "D":
        y.append(1)
    else:
        y.append(2)

# pad the sequences
max_len = max(len(x) for x in X)
X = [x + [0] * (max_len - len(x)) for x in X]

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# convert the data to tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# build the model
input_size = len(vocab)
hidden_size = 128
output_size = 3
model = LSTM(input_size, hidden_size, output_size)

# define the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# training loop
num_epochs = 10
batch_size = 2

for epoch in range(num_epochs):
    # shuffle the data
    permutation = torch.randperm(X_train.size()[0])
    X_train = X_train[permutation]
    y_train = y_train[permutation]
    
    # split the data into batches
    num_batches = X_train.size()[0] // batch_size
    for i in range(num_batches):
        # get the batch data
        X_batch = X_train[i*batch_size:(i+1)*batch_size]
        y_batch = y_train[i*batch_size:(i+1)*batch_size]
        
        # initialize the hidden state and cell state for the LSTM
        h_prev = torch.zeros((hidden_size, 1))
        c_prev = torch.zeros((hidden_size, 1))
        
        # zero the gradients of the model parameters
        optimizer.zero_grad()
        
        # forward pass
        h, c, y_pred = model(X_batch, h_prev, c_prev)
        
        # compute the loss
        loss = criterion(y_pred.t(), y_batch)
        
        # backward pass
        loss.backward()
        
        # update the model parameters
        optimizer.step()

# test the model
h_prev = torch.zeros((hidden_size, 1))
c_prev = torch.zeros((hidden_size, 1))
y_pred = model(X_test, h_prev, c_prev)
predictions = y_pred.argmax(dim=1)
accuracy = (predictions == y_test).float().mean()
print(f'Test accuracy: {accuracy:.2f}')






#To use an optimizer and prevent overfitting, you can follow these steps:
#Import the optimizer module from PyTorch, e.g. from torch import optim.
#Create an optimizer object, passing in the model's parameters as an argument, e.g. optimizer = optim.SGD(model.parameters(), lr=0.01).
#Before each training iteration, reset the gradients to zero by calling optimizer.zero_grad().
#After computing the loss, compute the gradients of the loss with respect to the model's parameters by calling loss.backward().
#Update the model's parameters by calling optimizer.step().
#You may also want to consider using techniques such as dropout or weight decay to prevent overfitting and improve the generalization of the model. You can do this by adding dropout layers to the model or setting the weight_decay argument when creating the optimizer.
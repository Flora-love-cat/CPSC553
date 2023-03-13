# ps3_functions.py
# CPSC 453 -- Problem Set 3
#
# This script contains pytorch shells for a feed forward network and an autoencoder.
#
from torch.nn.functional import softmax, relu, softplus, elu, tanh 
from torch import optim, nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch
import os
import scipy 
from pyemd import emd_samples

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class FeedForwardNet(nn.Module):
    """ Simple feed forward network with one hidden layer."""
    def __init__(self, activation="linear", input_size=784, 
                 output_size=10, hidden_size=128): # initialize the model
        super(FeedForwardNet, self).__init__() # call for the parent class to initialize
        # You can define variables here that apply to the entire model (e.g. weights, biases, layers...)
        # Here's how you can initialize the weight:
        # W = nn.Parameter(torch.zeros(shape)) # this creates a model parameter out of a torch tensor of the specified shape
        # ... torch.zeros is much like numpy.zeros, except optimized for backpropogation. We make it a model parameter and so it will be updated by gradient descent.
        # self.W1 = nn.Parameter(torch.tensor(np.random.uniform(low=-1/np.sqrt(10), high=1/np.sqrt(10), size=(784, 128)), dtype=torch.float), requires_grad=True)

        if activation == "linear":
          self.activation = nn.Identity()
        elif activation == "sigmoid":
          self.activation = nn.Sigmoid()
        elif activation == "relu":
          self.activation = nn.ReLU()
        elif activation == "softplus":
          self.activation = nn.Softplus()
        elif activation == "elu":
          self.activation = nn.ELU()
        elif activation == "tanh":
          self.activation = nn.Tanh()


        self.W1 = nn.Parameter(torch.zeros(input_size, hidden_size))
        torch.nn.init.uniform_(self.W1, -1/np.sqrt(hidden_size), 1/np.sqrt(hidden_size))
        
        # create a bias variable here
        self.b1 = nn.Parameter(torch.zeros(1, hidden_size))
        torch.nn.init.uniform_(self.b1, -1/np.sqrt(hidden_size), 1/np.sqrt(hidden_size))
        
        # Make sure to add another weight and bias vector to represent the hidden layer.
        self.W2 = nn.Parameter(torch.zeros(hidden_size, output_size))
        torch.nn.init.uniform_(self.W2, -1/np.sqrt(output_size), 1/np.sqrt(output_size))

        self.b2 = nn.Parameter(torch.zeros(1, output_size))
        torch.nn.init.uniform_(self.b2, -1/np.sqrt(output_size), 1/np.sqrt(output_size))

    def forward(self, x):
        """
        this is the function that will be executed when we call the feed-fordward network on data.
        INPUT:
            x, an MNIST image represented as a tensor of shape 784
        OUTPUT:
            predictions, a tensor of shape 10. If using CrossEntropyLoss, your model
            will be trained to put the largest number in the index it believes corresponds to the correct class.
        """
        
        # put the logic here. 
        # Be sure to add some type of nonlinearity to the output of the first layer, then pass it onto the hidden layer.
        H = self.activation(x @ self.W1 + self.b1) 
        
        predictions = H @ self.W2 + self.b2
        
        return predictions

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.en_lin1 = nn.Linear(784, 1000)
        # define additional layers here
        self.en_lin2 = nn.Linear(1000, 500)
        self.en_lin3 = nn.Linear(500, 250)
        self.en_lin4 = nn.Linear(250, 2)

        self.de_lin1 = nn.Linear(2, 250)
        self.de_lin2 = nn.Linear(250, 500)
        self.de_lin3 = nn.Linear(500, 1000)
        self.de_lin4 = nn.Linear(1000, 784)


    def encode(self, x):
        x = self.en_lin1(x)
        # ... additional layers, plus possible nonlinearities.
        x = F.tanh(x) 
        x = self.en_lin2(x)
        x = F.tanh(x)
        x = self.en_lin3(x)
        x = F.tanh(x)
        x = self.en_lin4(x) 

        return x

    def decode(self, z):
        # ditto, but in reverse
        z = self.de_lin1(z) 
        z = F.tanh(z)  
        z = self.de_lin2(z)
        z = F.tanh(z)
        z = self.de_lin3(z)
        z = F.tanh(z)
        z = self.de_lin4(z)  
        z = F.sigmoid(z) 

        return z  

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)




# initialize the model (adapt this to each model)
model = FeedForwardNet()
# initialize the optimizer, and set the learning rate
SGD = torch.optim.SGD(model.parameters(), lr = 0.01) # This is absurdly high.
# initialize the loss function. You don't want to use this one, so change it accordingly
loss_fn = torch.nn.CrossEntropyLoss()
batch_size = 128



def train(model,loss_fn, optimizer, train_loader, test_loader, filename= "checkpoint.pt", print_conf_matrix=False, num_epochs: int = 100):
    """
    This is a standard training loop, which leaves some parts to be filled in.
    INPUT:
    :param model: an untrained pytorch model
    :param loss_fn: e.g. Cross Entropy loss of Mean Squared Error.
    :param optimizer: the model optimizer, initialized with a learning rate.
    :param training_set: The training data, in a dataloader for easy iteration.
    :param test_loader: The testing data, in a dataloader for easy iteration.
    :param filename: name for saved model, default to "checkpoint.pt", 
    :param print_conf_matrix: whether to print a confusion matrix, deafult to False
    """

    train_loss, test_loss = np.zeros(num_epochs), np.zeros(num_epochs)
    best_test_acc = 0.
    best_epoch = None
    
    # num_epochs = 100 # obviously, this is too many. I don't know what this author was thinking.
    
    
    for epoch in range(num_epochs):
        # loop through each data point in the training set
        for data, targets in train_loader:
          data = data.to(device)
          targets = targets.to(device)
          
          optimizer.zero_grad()

          # run the model on the data
          model_input = data.reshape(-1, 784)# TODO: Turn the 28 by 28 image tensors into a 784 dimensional tensor.
          out = model(model_input)

          # Calculate the loss
          loss = loss_fn(out,targets)

          # Find the gradients of our loss via backpropogation
          loss.backward()

          # Adjust accordingly with the optimizer
          optimizer.step()       
        
        train_acc, train_loss[epoch] = evaluate(model, train_loader, loss_fn)
        test_acc, test_loss[epoch] = evaluate(model, test_loader, loss_fn)

        # Give status reports every 100 epochs
        if epoch % 5==0:
            print(f" EPOCH {epoch}. Progress: {epoch/num_epochs*100}%. ")
            print(f" Train accuracy: {train_acc}. Test accuracy: {test_acc}") #TODO: implement the evaluate function to provide performance statistics during training.

        
        # Save checkpoint 
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch 

            with timer("Saving checkpoint..."):
                torch.save(model.state_dict(), filename)

    print("The best validation accuracy of {:.3f} occurred after epoch {}."
          "".format(best_test_acc, best_epoch))

    model.load_state_dict(torch.load(filename))

    evaluate(model, test_loader, loss_fn, print_conf_matrix=print_conf_matrix) 

    plt.plot(range(num_epochs), train_loss, 'g', label="Train loss")
    plt.plot(range(num_epochs), test_loss, 'b', label="Test loss")
    plt.title("Train loss and test loss")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('feedforwardnet_mnist_train_test_loss.png') 
    plt.show()
    

    return model

def evaluate(model, evaluation_set, loss_fn, print_conf_matrix=False):
    """
    Evaluates the given model on the given dataset.
    Returns the percentage of correct classifications out of total classifications.
    """
    labels = []
    preds = []

    total_loss = np.zeros(len(evaluation_set.dataset))

    with torch.no_grad(): # this disables backpropogation, which makes the model run much more quickly.
        # TODO: Fill in the rest of the evaluation function.

        num_correct = 0
        for i, (data, targets) in enumerate(evaluation_set):
          data = data.to(device)
          targets = targets.to(device)
          # run the model on the data
          model_input = data.reshape(-1, 784)# TODO: Turn the 28 by 28 image tensors into a 784 dimensional tensor.
          out = model(model_input)
          pred = out.argmax(dim=-1)

          loss = loss_fn(out, targets)

          total_loss[i] = loss.item()

          num_correct += int((pred == targets).sum()) 
          
          labels.extend(targets.detach().cpu().numpy())
          preds.extend(pred.detach().cpu().numpy())
          
        accuracy = num_correct / len(evaluation_set.dataset)

        loss = np.mean(total_loss)

    if print_conf_matrix:
      print(confusion_matrix(labels, preds))

    return accuracy, loss


# ----- Functions for Part 5 -----
def mmd(X: np.ndarray, Y: np.ndarray, kernel_fn: callable, sigma: float=1):
    """
    Returns Maximum Mean Discrepancy of 2 distributions
    :param 
    X: samples from distribution 1. (N, d)
    Y: samples from distribution 2. (N, d)
    kernel_fn: a kernel function
    sigma: kernel bandwidth
    """
    mmd = np.mean(kernel_fn(X, X, sigma)) + np.mean(kernel_fn(Y, Y, sigma)) - 2*np.mean(kernel_fn(X, Y, sigma))
    return mmd

def kernel(X: np.ndarray, Y: np.ndarray, sigma: float=1) -> np.ndarray:
    """
    A gaussian kernel on two arrays.
    :param 
    X: samples from distribution 1. (N, d)
    Y: samples from distribution 2. (N, d)
    sigma: kernel bandwidth
    :return 
    K:  A symmetric matrix (N, N) in which k_{i,j} = e^{-||A_i - B_j||^2/(2*sigma^2)}
    """
    n = X.shape[0]
    K = np.zeros([n, n])
    for i in range(n):
      for j in range(n):
        K[i, j] = np.exp(-(np.linalg.norm(X[i] - Y[j])/sigma)**2/2)
    return K

def KLdivergence(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Returns KL Divergence of 2 distributions 
    :param 
    X: samples from distribution 1. (N, d)
    Y: samples from distribution 2. (N, d)
    """
    densityX = np.exp(KernelDensity(kernel='gaussian', metric='euclidean').fit(X).score_samples(X))  # np.histogram(X, bins=100, density=True)[0]
    densityY = np.exp(KernelDensity(kernel='gaussian', metric='euclidean').fit(Y).score_samples(Y))
    return scipy.stats.entropy(densityX, densityY)  


def EMD(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Returns earth mover's distance of 2 distributions
    :param 
    X: samples from distribution 1. (N, d)
    Y: samples from distribution 2. (N, d)
    """
    return emd_samples(X, Y, distance='euclidean')


# def EMD(p, q):
#     x = p - q
#     y = torch.cumsum(x, dim=0)
#     return y.abs().sum()

# load training set
mnist_train = datasets.MNIST(root='data',
                            train=True,
                            download=True,
                            transform=transforms.ToTensor()) 
# load test set 
mnist_test = datasets.MNIST(root='data',
                            train=False,
                            download=True,
                            transform=transforms.ToTensor()) 
                            
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)
 
train(model,loss_fn, SGD, train_loader, test_loader)
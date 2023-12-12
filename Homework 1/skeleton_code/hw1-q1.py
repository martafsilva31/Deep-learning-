#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        predictions = np.dot(self.W, x_i)
        y_hat = np.argmax(predictions)

        if y_hat != y_i:
            self.W[y_i] += x_i
            self.W[y_hat] -= x_i

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        
        label_scores = self.W.dot(x_i)
        e_y = np.zeros_like(label_scores)
        e_y[y_i] = 1

        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        self.W += learning_rate * np.outer(e_y - label_probabilities, x_i)

class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        mu, sigma = 0.1 , 0.1
        self.W1 = np.random.normal(mu, sigma, size = (hidden_size, n_features))
        self.W2 = np.random.normal(mu, sigma, size = (n_classes, hidden_size))

        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(n_classes)

        self.e_y = np.zeros(n_classes)
        
    # def softmax(self, x):
    #     exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    #     return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def softmax(self,x):
        exp_scores = np.exp(x - np.max(x))
        return exp_scores / np.sum(exp_scores)
    
    def forward(self, x_i):
        z = self.W1.dot(x_i) + self.b1
        h = np.maximum(0,z) #ReLU
        output = self.W2.dot(h) + self.b2
        probs = self.softmax(output)

        return probs, z, h
    
    def backward(self,x_i, y_i, probs ,z, h, learning_rate):
        self.e_y[y_i] = 1
        
        h1 = h
        h0 = x_i
        z1 = z
        
        grad_z2 = probs - self.e_y #gradient at the output layer
        grad_b2 = grad_z2
        grad_w2 = np.outer(grad_z2, h1)
        
        grad_h1 = self.W2.T.dot(grad_z2)
        grad_z1 = grad_h1*  (z1 > 0)
        grad_b1 = grad_z1
        grad_w1 = np.outer(grad_z1,h0)

        self.W1 -= learning_rate * grad_w1
        self.W2 -= learning_rate * grad_w2

        self.b1 -= learning_rate * grad_b1
        self.b2 -= learning_rate * grad_b2

        self.e_y[y_i] = 0
    

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        Y =[]
        for x_i in X:
            Y.append(np.argmax(self.forward(x_i)[0]))
        return Y

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):

        """
        Dont forget to return the loss of the epoch.
        """
        total_loss = 0
        for x_i, y_i in zip(X, y):
            probs, z, h = self.forward(x_i)
            self.backward(x_i,y_i,probs, z, h, learning_rate)
            
            loss = -np.log(probs[y_i])
            total_loss += loss
        
        return total_loss


def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()

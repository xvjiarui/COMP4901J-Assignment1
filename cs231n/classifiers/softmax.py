import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W) 
  for i in range(num_train):
    exp_sum = np.sum(np.exp(scores)[i, :])
    for j in range(num_class):
      cur_exp = np.exp(scores[i, y[i]])
      if j == y[i]:
        dW[:, j] -= (exp_sum - cur_exp)/exp_sum * X[i, :]
      else :
        dW[:, j] += np.exp(scores[i, j])/exp_sum * X[i, :]
    loss -= np.log(cur_exp/exp_sum)
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train 
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # N * C
  scores = X.dot(W)
  exp_sums = np.sum(np.exp(scores), axis = 1)
  correct_exp = np.exp(scores)[np.arange(num_train), y]
  loss = -np.sum(np.log(correct_exp/exp_sums))
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW = X.T.dot(np.exp(scores)/exp_sums.reshape((-1, 1)))
  binary = np.zeros_like(scores)
  binary[np.arange(num_train), y] = 1
  dW -= X.T.dot(binary)
  dW /= num_train 
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_samples = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_samples):
    x = X[i]
    scores = np.exp(np.dot(x, W))
    probility = scores / scores.sum()
    loss += -np.log(probility[y[i]])
    for j in range(num_class):
      if j != y[i]:
        dW[:, j] += -1/probility[y[i]]*(-probility[y[i]]*probility[j])*x
      else:
        dW[:, j] += -1/probility[j]*(-probility[j]**2 + probility[j])*x
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss = loss / num_samples + reg*np.sum(W*W)
  dW = dW/num_samples + reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_samples = X.shape[0]
  num_class = W.shape[1]
  scores = np.exp(np.dot(X, W))
  scores_reverse = 1.0/np.sum(scores, axis=1)
  probilities = scores_reverse.reshape(-1,1) * scores
  loss += -np.log(probilities[range(num_samples), y]).sum()
  probilities[range(num_samples), y] = probilities[range(num_samples), y] - 1
  dW = np.dot(X.T, probilities)
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss = loss / num_samples + reg * np.sum(W * W)
  dW = dW/num_samples + reg * W
  return loss, dW


from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params["W1"] = weight_scale*np.random.normal(0., 1.0, size=[input_dim, hidden_dim])
        self.params["b1"] = np.zeros(shape=(hidden_dim,))
        self.params["W2"] = weight_scale*np.random.normal(0., 1.0, size=[hidden_dim, num_classes])
        self.params["b2"] = np.zeros(shape=(num_classes, ))
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1 = self.params["W1"]
        b1 = self.params["b1"]
        W2 = self.params["W2"]
        b2 = self.params["b2"]
        hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
        scores = np.dot(hidden_layer, W2) + b2
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        num_samples, num_class = X.shape
        f = scores - np.max(scores, axis=1, keepdims=True)
        loss = -(f[range(num_samples), y] + np.log(np.sum(np.exp(f), axis=1, keepdims=True))).sum()
        loss = loss / num_samples + 0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2))
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        dscores = np.exp(f)/np.exp(f).sum(axis=1, keepdims=True)
        dscores[range(num_samples), y] -= 1
        dW1 = np.dot(hidden_layer.T, dscores)
        db1 = np.sum(dscores, axis=0)

        dhidden = np.dot(dscores, W2.T)
        dhidden[hidden_layer<=1e5] = 0
        dW2 = np.dot(X.T, dhidden)
        db2 = np.sum(dhidden, axis=0)
        grads["W1"] = dW1/num_samples + self.reg*W1
        grads["W2"] = dW2/num_samples + self.reg*W2
        grads["b1"] = db1/num_samples
        grads["b2"] = db2/num_samples
        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        last_layer_dim = input_dim
        for i in range(self.num_layers-1):
            self.params["W"+str(i+1)] = weight_scale*np.random.normal(0, 1.0, size=[last_layer_dim, hidden_dims[i]])
            self.params["b"+str(i+1)] = np.zeros(hidden_dims[i])
            if self.normalization:
                self.params["gamma"+str(i+1)] = np.ones(shape=(hidden_dims[i], ))
                self.params["beta"+str(i+1)] = np.zeros(shape=(hidden_dims[i], ))
            last_layer_dim = hidden_dims[i]
        self.params["W" + str(self.num_layers)] = weight_scale * np.random.normal(0, 1.0, size=[hidden_dims[-1],
                                                                                                     num_classes])
        self.params["b" + str(self.num_layers)] = np.zeros(shape=(num_classes,))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        N = X.shape[0]
        forward_layers = [{} for i in range(self.num_layers-1)]
        out = X
        for i in range(self.num_layers-1):
            out,cache = affine_forward(out, self.params["W"+str(i+1)], self.params["b"+str(i+1)])
            forward_layers[i]["affine"] = cache
            if self.normalization == "batchnorm":
                out,cache = batchnorm_forward(out, self.params["gamma"+str(i+1)], self.params["beta"+str(i+1)]
                                              , self.bn_params[i])
                forward_layers[i]["batchnorm"] = cache
            out, cache = relu_forward(out)
            forward_layers[i]["relu"] = cache
            if self.use_dropout:
                out, cache = dropout_forward(out, self.dropout_param)
                forward_layers[i]["dropout"] = cache
        scores = np.dot(out, self.params["W"+str(self.num_layers)]) + self.params["b"+str(self.num_layers)]

        # num_samples = X.shape[0]
        # forward_layers = {}
        # dropout_layers = {}
        # activation_layers={}
        # x_hat_layers = {}
        # mean_std = [{"mean":0,"std":0} for i in range(self.num_layers-1)]
        # last_layer = X
        # for i in range(self.num_layers-1):
        #     affine = np.dot(last_layer, self.params["W" + str(i + 1)]) + self.params["b" + str(i+1)]
        #     if self.normalization == "batchnorm":
        #         if self.bn_params[i]["mode"] == "train":
        #             mean = np.mean(affine, axis=0)
        #             std = np.std(affine, axis=0)
        #             affine_norm = (affine - mean)/std
        #             x_hat_layers[i] = affine_norm
        #             mean_std[i]["mean"] = mean
        #             mean_std[i]["std"] = std
        #             y_hat = self.params["gamma" + str(i + 1)]*affine_norm + self.params["beta" + str(i+1)]
        #         else:
        #             y_hat = self.params["gamma" + str(i+1)]*affine + self.params["beta" + str(i+1)]
        #     else:
        #         y_hat = affine
        #     relu = np.maximum(0, y_hat)
        #     activation_layers[i] = relu
        #     if self.use_dropout:
        #         if self.dropout_param["mode"] == "train":
        #             drop_indices = np.random.rand(*relu.shape)<self.dropout_param["p"]
        #             relu*=drop_indices
        #             dropout_layers[i] = drop_indices
        #         else:
        #             relu = self.dropout_param["p"] * relu
        #     forward_layers[i] = relu
        #     last_layer = relu
        # scores = np.dot(last_layer, self.params["W" + str(self.num_layers)]) + self.params["b"+str(self.num_layers)]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        f = scores - np.max(scores, axis=1, keepdims=True)
        loss = -(f[range(N), y]+np.log(np.exp(f).sum(axis=1))).sum()
        reg_loss = 0
        for i in range(self.num_layers):
            reg_loss += np.sum(self.params["W"+str(i+1)]*self.params["W"+str(i+1)])
        loss = loss/N + 0.5*self.reg*reg_loss

        dout = np.exp(scores)/np.sum(np.exp(scores), axis=1, keepdims=True)
        dout[range(N), y] -= 1

        for i in range(self.num_layers-2, -1, -1):
            dout, dw, db = affine_backward(dout, forward_layers[i]["affine"])
            grads["W"+str(i+1)] = dw
            grads["b"+str(i+1)] = db
            if self.use_dropout:
                dout = dropout_backward(dout, forward_layers[i]["dropout"])
            dout = relu_backward(dout, forward_layers[i]["relu"])
            if self.normalization == "batchnorm":
                dout, dgamma, dbeta = batchnorm_backward(dout, forward_layers[i]["batchnorm"])
                grads["gamma"+str(i+1)] = dgamma
                grads["beta"+str(i+1)] = dbeta


        # for i in range(self.num_layers-2, -1, -1):
        #     grads["W"+str(i+1)] = np.dot(forward_layers[i].T, dout)
        #     grads["b" + str(i+1)] = np.sum(dout, axis=0)
        #     relu_diff = np.dot(dout, self.params["W"+str(i+1)].T)*dropout_layers[i]
        #     y_hat_diff = relu_diff*((activation_layers[i]>0).astype(np.int))
        #     if self.normalization == "batchnorm":
        #         if self.bn_params[i]['mode'] == "train":
        #             x_hat_diff = y_hat_diff*self.bn_params[i]["gamma"+str(i+1)]
        #             grads["gamma"+str(i+1)] = np.sum(y_hat_diff * x_hat_layers[i], axis=0)
        #             grads["beta"+str(i+1)] = np.sum(y_hat_diff, axis=0)
        #             tmp1 = np.sum(x_hat_diff, axis=0, keepdims=True)
        #             tmp2 = (1 + x_hat_layers[i]*x_hat_layers[i])/(N*np.sqrt(self.bn_params[i]["std"]*self.bn_params[i]["std"]+1e-10))
        #             affine_diff = x_hat_diff/np.sqrt(self.bn_params[i]["std"]*self.bn_params[i]["std"]+1e-10) \
        #             - tmp1*tmp2
        #             dout = affine_diff
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return loss, grads

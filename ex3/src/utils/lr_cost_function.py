import numpy as np
from .sigmoid import sigmoid


def lr_cost_function(theta, x, y, lam, alpha=1):
    """
    Logistic Regression Cost Function.

    Compute cost and gradient for logistic regression with regularization

    m = size(y)
    cost = 1/m * (sum(-y * log(h_x) - (1-y) * log(1-h_x))) + lambda * sum(theta^2))

    k = size(theta)
    regularized_gradient = [grad1, grad2, ... grad_k]

    :param theta: theta parameters of the model
    :param x: training set
    :param y: training set labels
    :param lam: lambda for regularization
    :param alpha: alpha parameter for gradient

    :return: (cost, gradient) for the given parameters of the model
    """

    m = np.size(y)

    """
    ================================================ YOUR CODE HERE ====================================================
    Instructions: Compute the cost of a particular choice of theta. Compute the partial derivatives and set grad to the
                  partial derivatives of the cost w.r.t. each parameter in theta.

    Hint: The computation of the cost function and gradients can be efficiently vectorized.
          For example, consider the following computation:
    
          ```
          h_x = sigmoid(np.matmul(x, theta))
          ```
    
          Each row of the resulting matrix will contain the value of the prediction for that example.
          You can make use of this to vectorize the cost function and gradient computations.

    Hint: Computing the regularized gradient can be done the following way:
          ```
          grad = <NOT REGULARIZED GRADIENT>
          tmp = theta
          tmp[0] = 0
          grad_reg = <YOUR CODE>

          grad = grad + grad_reg
          ```
    """

    # Hypothesis (z)
    hx = np.matmul(x, theta)
    hx = sigmoid(hx)

    # Regularization
    # excluding the bias (theta[0])
    theta_tmp = theta
    theta_tmp[0] = 0
    reg = lam/2 * np.sum(theta_tmp ** 2)

    # Cost
    cost = np.sum(-1 * y * np.log(hx) - (1 - y) * np.log(1 - hx))
    cost = (cost + reg) / m

    # Calculate gradients
    grad = np.dot(x.T, hx - y)
    grad = grad / m
    grad = grad * alpha

    # adding regularization to gradient
    grad_reg = lam/m * theta_tmp

    grad = grad + grad_reg

    return cost, grad


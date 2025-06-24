import numpy as np

def cost(theta, X, y):
    c = 0
    for i in range(X.shape[0]):
        c += (np.dot(X[i], theta) - y[i]) ** 2
    c = c / (2 * X.shape[0])
    return c  

def gradient_compute(theta, X, y):
    m, n = X.shape
    grad = np.zeros_like(theta)  
    for i in range(n):
        c = 0
        for j in range(m):
            c += (np.dot(X[j], theta) - y[j]) * X[j, i]  
        grad[i] = c / m
    return grad

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m, n = X.shape
    theta = np.zeros((n, 1))
    y = y.reshape(m, 1)  
    for i in range(iterations):
        grad = gradient_compute(theta, X, y)
        theta = theta - alpha * grad
    return np.round(theta, 4)
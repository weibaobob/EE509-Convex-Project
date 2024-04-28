import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 0.5 * np.sum(x**2)

def grad_f(x):
    return x

def backtracking_line_search(x, grad, f, alpha=0.5, beta=0.1):
    t = 1.0  # Start with a step size of 1
    while f(x - t * grad) > f(x) - alpha * t * np.dot(grad, grad):
        t *= beta
    return t

#Apply GD with backtracking 
def gradient_descent(initial_point, num_iterations, f, grad_f):
    x = initial_point
    path = [x.copy()]  # to store the path of x
    for i in range(num_iterations):
        current_grad = grad_f(x)
        step_size = backtracking_line_search(x, current_grad, f)
        x -= step_size * current_grad
        path.append(x.copy())
        print(f"GD Iteration {i+1}: x = {x}, Step Size = {step_size}, Function Value = {f(x)}")
    return np.array(path)

#Apply SGD with backtracking 
def stochastic_gradient_descent(initial_point, num_iterations, f, grad_f):

    x = initial_point
    path = [x.copy()]
    for i in range(num_iterations):
        current_grad = grad_f(x)
        step_size = backtracking_line_search(x, current_grad, f)
        x -= step_size * current_grad
        path.append(x.copy())
        print(f"SGD Iteration {i+1}: x = {x}, Step Size = {step_size}, Function Value = {f(x)}")
    return np.array(path)

initial_point = np.array([-1.0, 2.0])  # Starting point
num_iterations = 3                  # Number of iterations

# Run both methods
trajectory_gd = gradient_descent(initial_point, num_iterations, f, grad_f)
trajectory_sgd = stochastic_gradient_descent(initial_point, num_iterations, f, grad_f)

# Plotting 
plt.figure(figsize=(8, 6))
plt.plot(trajectory_gd[:, 0], trajectory_gd[:, 1], 'o-', label='Gradient Descent')
plt.plot(trajectory_sgd[:, 0], trajectory_sgd[:, 1], 's-', label='Stochastic Gradient Descent')
plt.title('Comparison of GD and SGD on $f(x) = 1/2 ||x||^2$')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.show()

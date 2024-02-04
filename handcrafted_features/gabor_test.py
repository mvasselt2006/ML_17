from math import exp, cos, sin, pi

def g(x, y, lamda, theta, psi, sigma, gamma):
    
    x_ =  x * cos(theta) + y * sin(theta)
    y_ = -x * sin(theta) + y * cos(theta)
    
    return exp(-(x_ ** 2 + (gamma * y_) ** 2) / (2 * sigma ** 2)) * cos(2 * pi * x_ / lamda + psi)

sigma = 2
psi   = 0
gamma = 0.5
lamda = 10

x = 1
y = 0
theta = 0

print(g(x, y, lamda, theta, psi, sigma, gamma))
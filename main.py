import numpy as np

class Node:
    def __init__(self, value, requires_grad = True):
        self.value = value
        self.gradient = np.zeros_like(value, dtype=float) # vector of 1s the same size as the value vector
        self.parents = []
        self.requires_grad = requires_grad

    def __add__(self, other):
        if isinstance(other, Node): # check if other is a Node object to use value attribute
            out = Node(self.value + other.value)
            out.parents = [(self, 1), (other, 1)] # d/dx (u + v) = (1, 1) for each var
        else:
            out = Node(self.value + other)
            out.parents = [(self, 1)]
        return out
        
    def __mul__(self, other):
        if isinstance(other, Node): # check if other is a Node object to use value attribute
            out = Node(self.value * other.value)
            out.parents = [(self, other.value), (other, self.value)]
        else:
            out = Node(self.value * other)
            out.parents = [(self, other)]
        return out
    
    def __pow__(self, n):
        out = Node(self.value**n)
        out.parents = [(self, n*self.value**(n-1))]
        return out
    
    # handling trig and other common functions
    def sin(self):
        out = Node(np.sin(self.value))
        out.parents = [(self, np.cos(self.value))]
        return out
    
    def cos(self):
        out = Node(np.cos(self.value))
        out.parents = [(self, -np.sin(self.value))]
        return out
    
    def log(self):
        out = Node(np.log(self.value))
        out.parents = [(self, 1 / self.value)]
        return out
    
    def exp(self):
        out = Node(np.exp(self.value))
        out.parents = [(self, np.exp(self.value))]
        return out

    def backward(self, grad):
        if(self.requires_grad):
            self.gradient += grad
            for parent, local_grad in self.parents:
                parent.backward(grad * local_grad)
    
u = Node(np.array([1, 2, 5]), requires_grad=True)
z = Node(np.array([2, 5, -1]), requires_grad=True)
h = u*(u+5)**2 + z**3 # updates u and z's gradients
g = h.cos() # updates h's gradient
f = g**2 # updates g's gradient
f.backward(grad=np.ones_like(f.value)) # starts backpropagation
dfdu = u.gradient
dfdz = z.gradient
gradient = np.array([dfdu, dfdz])
gradient

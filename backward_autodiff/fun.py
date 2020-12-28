import torch
import numpy as np

from scipy.special import expit 

def pprint(array):
    print(np.round(array, 4))

def torch_test(X_shape=(5, 3), w_shape=(3, 1), seed=42):
    np.random.seed(seed)
    
    X = torch.tensor(np.random.uniform(size=X_shape), requires_grad=True)
    w = torch.tensor(np.random.uniform(size=w_shape), requires_grad=True)
    
    z = (X @ w)
    z.retain_grad()
    L = torch.sigmoid(z)
    L.retain_grad()
    
    L.backward(torch.ones_like(L))
    
    print()
    print(f"L shape: {L.shape}")
    print(f"z shape: {z.shape}")
    print(f"X shape: {X.shape}")
    print(f"w shape: {w.shape}")
    
    print()
    print(L.grad.shape)
    print(z.grad.shape)
    
    print(f"X grad shape: {X.grad.shape}")
    print(X.grad)
    print(f"w grad shape: {w.grad.shape}")
    print(w.grad)
    

def torch_test_vector():
    torch_test(X_shape=(5, 3), w_shape=(3, 1), seed=42)
    
def torch_test_matrix():
    torch_test(X_shape=(5, 3), w_shape=(3, 4), seed=42)
    
    
def my_test_vector(X_shape=(5, 3), w_shape=(3, 1), seed=42):
    np.random.seed(seed)
    
    X = np.random.uniform(size=X_shape)
    w = np.random.uniform(size=w_shape)
    
    z = X @ w
    L = expit(z)
    
    dLdz = np.identity(L.shape[0]) * (expit(z) * (1 - expit(z)))
    
    dzdX = np.tile(w.T, (X.shape[0], 1))
    dzdW = X
    
    X_grad = dLdz @ dzdX
    w_grad = (dLdz @ dzdW).sum(0).reshape(-1, 1)
    
    print(f"X grad shape: {X_grad.shape}")
    pprint(X_grad)
    print(f"w grad shape: {w_grad.shape}")
    pprint(w_grad)

    # print(np.round(L, 4))
    
def my_test_matrix(X_shape=(5, 3), w_shape=(3, 1), seed=42):
    np.random.seed(seed)
    
    X = np.random.uniform(size=X_shape)
    w = np.random.uniform(size=w_shape)
    
    z = X @ w
    L = expit(z)
    
    dLdz = expit(z) * (1 - expit(z))
    

    X_grad = dLdz @ w.T
    w_grad = X.T @ dLdz
    
    print(f"X grad shape: {X_grad.shape}")
    pprint(X_grad)
    print(f"w grad shape: {w_grad.shape}")
    pprint(w_grad)
    
    
if __name__ == "__main__":
    torch_test_vector()
    # print("-"*20)
    # my_test_vector() 
    # torch_test_matrix()
    my_test_matrix()
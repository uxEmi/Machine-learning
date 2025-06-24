import numpy as np

iterations = 10000
alpha = 0.001

def cost (x,y,m,b):
    c = 0
    for i in range(len(x)):
        c += (m * x[i] + b - y[i]) ** 2 
    c = 1 / (2 * len(x)) * c
    return c
def gradient_r(x,y,m,b):
    dm = db = 0
    for i in range(len(x)):
        dm += (m * x[i] + b - y[i]) * x[i] 
        db += m * x[i] + b - y[i] 
    dm = dm / len(x)
    db = db / len(x)
    return dm , db
def gradient_descendent(x,y,m,b):
    for i in range(iterations):
        ma,ba = gradient_r(x,y,m,b)
        
        m = m - alpha * ma
        b = b - alpha * ba
        print(f"b{b}  m{m}")
    return m,b 
x = np.array([1.0,1.5,2.0,2.3])
y = np.array([100000,150000,200000,230000])

m,b = gradient_descendent(x,y,0,0)

aux_x = float(input("Inrodu variabila"))

print(f"Pretul este : {m * aux_x + b}")
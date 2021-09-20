import numpy as np

# initializtaion loading weights 
def initilization():
    params = {}

    # initilization for conv
    params['W1'] = np.random.randn(4,4)  * 0.1
    params['b1' ] = np.zeros((1,1))
    params['W2' ] = np.random.randn(2,2)  * 0.1
    params['b2' ] = np.zeros((1, 1))

    #xavier initiliazer for dense layer
    params['W3' ] = np.random.randn(50,144) *   np.sqrt(2/144)
    params['b3' ] = np.zeros((50,1))
    params['W4' ] = np.random.randn(1,50) *   np.sqrt(2/50)
    params['b4' ] = np.zeros((1))
    return params


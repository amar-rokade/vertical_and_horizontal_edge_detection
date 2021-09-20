import numpy  as np
from layers import conv_forward,model_forward


#dataset making
def make_datasets() :
    X =  np.random.randint(255,size=(1000,8,8))
    # horizontal
    for i in range(X.shape[0]//2):
        b = np.zeros((np.random.randint(2,6),8))
        X[i,:b.shape[0],:] = b
    #vertical zeros set
    for i in range(X.shape[0]//2,X.shape[0]):
        b = np.zeros((8,np.random.randint(2,6)))
        X[i,:,:b.shape[1]] = b

    #zeros as HORIZONAL AND 1 AS VERTICAL
    Y = np.ones((1000,1))
    Y[:500] = 0
    # we shuffle the datasets
    p = np.random.permutation(len(Y))
    X_shuffled=X[p]
    Y_shuffled = Y[p]

    # we return the shuffled datasets
    return X_shuffled,Y_shuffled





#updating params
def update_params(params, grad, learning_rate):

    for l in range(1,(len(params)//2)+1):
        params['W' + str(l)] = params['W' + str(l)] - learning_rate * grad['dW' + str(l)]
        params['b' + str(l)] = params['b' + str(l)] - learning_rate * grad['db' + str(l)]
    return params

    #____________________________________________________________________________________________________________________



def predict(X,params):
    #conv layer forward
    Z1,_ = conv_forward(X,params["W1"],params["b1"],stride=1,pad=2)
    Z2,_ = conv_forward(Z1,params["W2"],params["b2"],stride=1,pad=2)

    #input for dense layer
    Z2_flatten = np.reshape(Z2,newshape=(Z2.shape[0],-1))

    #dense layer forward
    cache3 = model_forward(Z2_flatten,params)
    Y_pred  = cache3['A4']

    #zeros as HORIZONAL AND 1 AS VERTICAL
    for i in range(X.shape[0]):
        if Y_pred[i] > 0.5 :
            Y_pred[i] = 1
        else :
            Y_pred[i] = 0
    return Y_pred

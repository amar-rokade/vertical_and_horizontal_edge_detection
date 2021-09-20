import numpy as np

# when for forwad and back workd prop to not lose edge info
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0,0), (pad,pad),(pad,pad)),'constant', constant_values = 0)
    return X_pad



#e perform singpe step of convn
def conv_single_step(a_slice_prev,W,b):

    # single conv step
    Z = a_slice_prev * W + b
    Z = np.sum(Z)
    return Z


# forward prop
def conv_forward(A_prev,W,b,stride,pad):

    m,n_H_prev,n_W_prev = A_prev.shape
    f,f = W.shape
    #output n_H,n_W
    n_H  = ((n_H_prev - f + (2 * pad)) // stride) + 1
    n_W  = ((n_W_prev - f + (2 * pad)) // stride) + 1

    # output volume
    Z = np.zeros((m,n_H,n_W))

    #pad
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        #get particular example
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):

                # Starting points
                vertical_start = h * stride
                vertical_end =  vertical_start + f
                horizonatal_start = w * stride
                horizontal_end = horizonatal_start + f
                a_slice_prev = a_prev_pad[vertical_start:vertical_end,horizonatal_start:horizontal_end]
                # append value
                Z[i, h, w] = conv_single_step(a_slice_prev,W,b)
    # return cache
    cache = ( W, b)

    return Z,cache




#forward_propagations-----------------------------------------------
def forward_activation(A_prev, w, b, activation):

    #activation
    z = np.dot(A_prev, w.T) + b.T + pow(10,-4)
    if activation == 'relu':
        A = np.maximum(0, z)
    elif activation == 'sigmoid':
        A = 1/(1+np.exp(-z))
    else :
        A = np.tanh(z)
    return A



# dense two layer after conv2
def model_forward(X,params):
    cache = {}
    A3 = forward_activation(X, params['W3'], params['b3'], 'sigmoid')
    A4 = forward_activation(A3, params['W4'], params['b4'], 'sigmoid')
    #print('A3:',A3,'   A4:',A4)
    cache['A3'] = A3
    cache['A4'] = A4
    return cache





#____________________________________________________________________________________________________________________
#dense backward function
def backward(Y,X,params, cache):
    grad ={}
    m = Y.shape[0]

    dZ4 = cache['A4'] - Y
    grad['dW4'] = (1 / m) * np.dot(dZ4.T, cache['A3'])
    grad['db4'] = 1 / m * np.sum(dZ4.T, axis=1, keepdims=True)

    dZ3 = np.dot(dZ4, params['W4']) * (1 - np.power(cache['A3'], 2))
    #print(dZ3.shape,"cache['A3']",cache['A3'].shape)
    grad['dW3'] = (1 / m) * np.dot(dZ3.T, X)
    grad['db3'] = (1 / m) * np.sum(dZ3.T, axis=1, keepdims=True)
    #
    dZ2 = np.dot(dZ3, params['W3']) * (1 - np.power(X, 2))


    return grad, dZ2





#backprop for conv2
def conv_backward(dZ, W, b, A_prev, stride, pad):
    (m, n_H_prev, n_W_prev) = A_prev.shape

    (f, f) = W.shape

    (m, n_H, n_W) = dZ.shape

    dA_prev = np.zeros((m, n_H_prev, n_W_prev))
    dW = np.zeros((f, f))
    db = np.zeros((1, 1,))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                #starting point
                vert_start = h
                vert_end = vert_start + f
                horiz_start = w
                horiz_end = horiz_start + f

                a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end]

                da_prev_pad[vert_start:vert_end, horiz_start:horiz_end] += (W * dZ[i, h, w])
                #print(da_prev_pad.shape)
                dW += a_slice * dZ[i, h, w]
                db += dZ[i, h, w]


        dA_prev[i, :, :] = da_prev_pad[pad:-pad, pad:-pad]


    return dA_prev, dW, db

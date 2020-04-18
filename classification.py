import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams.update({'font.size': 20, 'figure.figsize': (10, 8)}) # To make plot larger

df = pd.read_csv("iris.data")

def change_label(str):
    if(str == "Iris-setosa"):
        return 0
    if(str == 'Iris-versicolor'):
        return 1
    return 2

df["species"] = df["species"].apply(change_label) #Changing species names to labels
df = df[df["species"]!=2] #classifying into only two labels

xvals = df[["sepal_length","sepal_width","petal_length","petal_width"]].values.T
yvals = df["species"].values.T
m = xvals.shape[1]
alpha = 1.5

def init_weights(features_size, hidden_size, output_size):
    np.random.seed(3)

    #Initializing weights and biases of both layers
    w1 = np.random.randn(hidden_size, features_size) * 0.01
    b1 = np.zeros((hidden_size,1))
    w2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size,1))

    params = {"w1" : w1, "b1" : b1, "w2" : w2, "b2" : b2}

    return params


def forward_propagation(params):
    w1 = params["w1"]
    w2 = params["w2"]
    b1 = params["b1"]
    b2 = params["b2"]

    z1 = np.dot(w1,xvals) + b1
    a1 = 1/(1+np.exp(-z1)) #Hidden layer
    z2 = np.dot(w2,a1) + b2
    a2 = 1/(1+np.exp(-z2)) #Output Layer
     
    act_vals = {"z1" : z1, "a1" : a1, "z2" : z2, "a2" : a2}
    return act_vals


def cost_function(output_layer, output_vals):
    a = np.multiply(np.log(output_layer),output_vals) + np.multiply((1-output_vals),np.log(1-output_layer))
    cost = -np.sum(a)/m
    return cost

def backward_propagation(params, act_vals):
    w1 = params["w1"]
    w2 = params["w2"]
    b1 = params["b1"]
    b2 = params["b2"]
    a1 = act_vals["a1"]
    a2 = act_vals["a2"]

    dz2 = a2 - yvals
    dw2 =  np.dot(dz2, a1.T) / m
    db2 =  np.sum(dz2, axis = 1, keepdims = True) / m
    dz1 = np.multiply(np.dot(w2.T,dz2), 1 - a1**2) 
    dw1 = np.dot(dz1, xvals.T) / m
    db1 = np.sum(dz1, axis = 1, keepdims = True) / m

    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2

    return params


def run_model():
    params = init_weights(4,6,1) #Taking one hidden layer with 6 nodes

    w1 = params["w1"]
    b1 = params["b1"]
    w2 = params["w2"]
    b1 = params["b2"]

    iterations = 10000
    k = iterations / 10

    for i in range(iterations):
        act_vals = forward_propagation(params)
        curr_cost = cost_function(act_vals["a2"],yvals)
        params = backward_propagation(params,act_vals)

        if i % k == 0:
            print("Cost after iterations %i : %f" % (i,curr_cost))

    return act_vals["a2"]


predictions = run_model()
predictions = predictions[[0],:].transpose()

def success_rate(predictions):
    cnt = 0

    for i in range (m):
        if predictions[i] >= 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0

    for i in range(m):
        if predictions[i,0] == yvals[i]:
            cnt = cnt + 1

    return cnt * 100 / m

print(success_rate(predictions))
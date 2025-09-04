import numpy as np
from matplotlib import pyplot as plt

# Define layers and layer sizes (2 inputs, 3 perceptrons, 3 perceptrons, 1 output)
Layers = 3
Perceptrons = [2, 3, 3, 1] 

# Initialize weights as random numbers 
rng = np.random.default_rng(seed = 42)
W1 = rng.normal(loc=0.0, scale=1.0, size=(Perceptrons[1], Perceptrons[0]))
W2 = rng.normal(loc=0.0, scale=1.0, size=(Perceptrons[2], Perceptrons[1]))
W3 = rng.normal(loc=0.0, scale=1.0, size=(Perceptrons[3], Perceptrons[2]))

# Initialize bias for each layer
b1 = rng.normal(loc=0.0, scale=1.0, size=(Perceptrons[1], 1)) # 1 bias for each neuron in first layer 
b2 = rng.normal(loc=0.0, scale=1.0, size=(Perceptrons[2], 1)) 
b3 = rng.normal(loc=0.0, scale=1.0, size=(Perceptrons[3], 1)) 

###############################
# Data generation
def generate_data(n):
    """
    Set the random seed
    Generate X as an array of values, the size is (n, 2) (number of observations, with 2 columns)
    Generate y as having a relationship with X
    Return X, y
    """
    rng = np.random.default_rng(seed = 42)
    X = rng.uniform(low = -1.0, high = 1.0, size = (n, 2))

    y_cont = (0.3*(X[:,1])**2 - (1/(1 - np.exp((X[:,0]*0.50 + X[:,1]*-1.50)))) - (1/(1 - np.exp((X[:,0]*1.40 + X[:,1]*-2.50)))) + (1/(1 - np.exp((X[:,0]*-2.0 + X[:,1]*0.25)))))
    y = (y_cont > np.median(y_cont)).astype(int)
    y = y.reshape(1,n)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    return X, y

##########################
# Activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Feed forward
def feed_forward(X):
    """
    Z = Calculate weighted sum for neurons (returns 3xn) 
    A = Apply sigmoid activation function
    """
    # First layer
    A0 = X.T
    Z1 = W1 @ A0 + b1
    A1 = sigmoid(Z1)

    # Second layer 
    Z2 = W2 @ A1 + b2 
    A2 = sigmoid(Z2)

    # Third layer 
    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    # Predict
    y_hat = (A3 > 0.50).astype(int)

    L_values = {
        "A0": A0,
        "A1": A1,
        "A2": A2,
        "A3": A3,
        "y_hat": y_hat
    }

    return L_values

########################
# Define cost 
def entropy(prob, y):
    """
    Cross entropy loss, summed across all subjects
    """
    losses = -( (y * np.log(prob)) + (1 - y)*(np.log(1 - prob)) )

    summed_loss = np.mean(losses)

    return summed_loss

#######################
# Backpropogation for layer 3
def backprop_layer_3(A2, A3, y, W3):
    m = A3.shape[1]
    # derivative of final layer output wrt cost
    dC_dZ3 = (A3 - y) / m

    # derivative of layer weights wrt cost
    dC_dW3 = dC_dZ3 @ A2.T

    # derivative of bias wrt cost
    dC_db3 = np.sum(dC_dZ3, axis=1, keepdims=True)

    # derivative of inputs wrt cost (for backprop of next layer)
    dC_dA2 = W3.T @ dC_dZ3

    return dC_dW3, dC_db3, dC_dA2

# Backprop for layer 2
def backprop_layer_2(dC_dA2, A1, A2, W2):
    m = A2.shape[1]
    # derivative of layer 2 output wrt cost
    dA2_dZ2 = A2 * (1 - A2)
    dC_dZ2 = dC_dA2 * dA2_dZ2

    # derivative of layer 2 weights wrt cost
    dC_dW2 = dC_dZ2 @ A1.T

    # derivative of bias wrt cost
    dC_db2 = np.sum(dC_dZ2, axis=1, keepdims=True)

    # derivative of layer 2 inputs wrt cost
    dC_dA1 = W2.T @ dC_dZ2

    return dC_dW2, dC_db2, dC_dA1

def backprop_layer_1(dC_dA1, A1, A0, W1):
    m = A1.shape[1]
    # derivative of layer 1 outputs wrt cost
    dA1_dZ1 = A1 * (1 - A1)
    dC_dZ1 = dC_dA1 * dA1_dZ1

    # derivative of layer 1 weights wrt cost
    dC_dW1 = dC_dZ1 @ A0.T

    # derivative of bias wrt cost
    dC_db1 = np.sum(dC_dZ1, axis=1, keepdims=True)

    return dC_dW1, dC_db1

######################
# To be added to eventual training function
def train(epochs, alpha):
    """
    epochs = iterations of GD
    alpha = learning rate

    Run forward pass, calculate loss, perform backprop, update weights based on gradient
    """
    global W1, W2, W3, b1, b2, b3, X, y
    cost = []

    for e in range(epochs):

        # FORWARD PASS 
        L_values = feed_forward(X)

        # CALCULATE COST 
        loss = entropy(prob = L_values['A3'], y=y)
        cost.append(loss)

        # BACKPROP x3 
        dC_dW3, dC_db3, dC_dA2 = backprop_layer_3(L_values["A2"], L_values["A3"], y, W3)
        dC_dW2, dC_db2, dC_dA1 = backprop_layer_2(dC_dA2, L_values["A1"], L_values["A2"], W2)
        dC_dW1, dC_db1 = backprop_layer_1(dC_dA1, L_values["A1"], L_values["A0"], W1)

        # UPDATE WEIGHTS 
        W3 = W3 - (alpha * dC_dW3)
        W2 = W2 - (alpha * dC_dW2)
        W1 = W1 - (alpha * dC_dW1)

        # UPDATE BIAS
        b1 = b1 - (alpha * dC_db1)
        b2 = b2 - (alpha * dC_db2)
        b3 = b3 - (alpha * dC_db3)

        # PRINT STATEMENT 
        if e % (epochs // 10) == 0:
            print(f"Cost for epoch {e}: {loss:3f}")

    # Define output
    model = {
        "cost": cost,
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "A0": L_values["A0"],
        "A1": L_values["A1"],
        "A2": L_values["A2"],
        "A3": L_values["A3"],
        "y_hat": L_values["y_hat"]
    }

    print("")
    print(f"Final Cost: {model['cost'][-1]}")
    print(f"Final W1: \n{model['W1']}")
    print(f"Final W2: \n{model['W2']}")
    print(f"Final W3: \n{model['W3']}")

    return model


######################
######################
n = 500 
epoch = 50000
alpha = 0.025

# Generate data
X, y = generate_data(n)

# Train
final_model = train(epochs=epoch, alpha=alpha)



#######################
print_unlabeled = 0
#######################
if print_unlabeled == 1:
    plt.scatter(final_model['A0'][0,:], final_model['A0'][1,:], s=60)
    plt.title("Original X1 and X2")
    plt.show()

    plt.scatter(final_model['A1'][0,:], final_model['A1'][1,:], s=60)
    plt.title("First representations")
    plt.show()

    plt.scatter(final_model['A2'][0,:], final_model['A2'][1,:], s=60)
    plt.title("Second representations")
    plt.show()

    plt.scatter(x=final_model['A3'], y=np.zeros_like(final_model['A3']), s=5)
    plt.title("Final 1-D representation")
    plt.show()

#######################
colors = np.array(["C0", "C1"])
print_labeled = 0
#######################
if print_labeled == 1:
    plt.scatter(final_model['A0'][0,:], final_model['A0'][1,:], c=colors[y.squeeze()], s=60)
    plt.title("Original X1 and X2")
    plt.show()

    plt.scatter(final_model['A1'][0,:], final_model['A1'][1,:], c=colors[y.squeeze()], s=60)
    plt.title("First representations of X1 and X2")
    plt.show()

    plt.scatter(final_model['A2'][0,:], final_model['A2'][1,:], c=colors[y.squeeze()], s=60)
    plt.title("Second representations of X1 and X2")
    plt.show()

    plt.scatter(x=final_model['A3'], y=np.zeros_like(final_model['A3']), c=colors[y.squeeze()], s=5)
    plt.title("Final 1-D representation")
    plt.show()

#######################
print_3D = 0
#######################
if print_3D == 1:
    plt.scatter(final_model['A0'][0,:], final_model['A0'][1,:], c=colors[y.squeeze()], s=60)
    plt.title("Original X1 and X2")
    plt.show()

    # First layer of 3 neurons (3 representations)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(final_model['A1'][0,:], final_model['A1'][1,:], final_model['A1'][2,:], c=colors[y.squeeze()])
    plt.show()

    # Second layer of 3 neurons (3 representations)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(final_model['A2'][0,:], final_model['A2'][1,:], final_model['A2'][2,:], c=colors[y.squeeze()])
    plt.show()

    plt.scatter(x=final_model['A3'], y=np.zeros_like(final_model['A3']), c=colors[y.squeeze()], s=5)
    plt.title("Final 1-D representation")
    plt.axvline(x=0.50, color='lightgray', linestyle='--', ymin=0.2, ymax=0.8)
    plt.show()


#######################
print_3D_star = 1
#######################
if print_3D_star == 1:

    star_data = final_model.copy()
    for key in ["A0", "A1", "A2", "A3"]:
        star_data[key] = np.delete(star_data[key], 5, axis=1)

    y_star = y.copy()
    y_star = np.delete(y_star, 5, axis=1)

    plt.scatter(star_data['A0'][0,:], star_data['A0'][1,:], c=colors[y_star.squeeze()], s=60)
    plt.scatter(final_model['A0'][0,5], final_model['A0'][1,5], s=150, marker='*', c='crimson')
    plt.title("Original X1 and X2")
    plt.show()

    # First layer of 3 neurons (3 representations)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(star_data['A1'][0,:], star_data['A1'][1,:], star_data['A1'][2,:], c=colors[y_star.squeeze()])
    ax.scatter(final_model['A1'][0,5], final_model['A1'][1,5], final_model['A1'][2,5], s=150, marker='*', c='crimson')
    plt.show()

    # Second layer of 3 neurons (3 representations)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(final_model['A2'][0,:], final_model['A2'][1,:], final_model['A2'][2,:], c=colors[y.squeeze()])
    ax.scatter(final_model['A2'][0,5], final_model['A2'][1,5], final_model['A2'][2,5], s=150, marker='*', c='crimson')
    plt.show()

    plt.scatter(x=final_model['A3'], y=np.zeros_like(final_model['A3']), c=colors[y.squeeze()], s=5)
    plt.scatter(x=final_model['A3'][0,5], y=0, s=150, marker='*', c='crimson')
    plt.title("Final 1-D representation")
    plt.axvline(x=0.50, color='lightgray', linestyle='--', ymin=0.2, ymax=0.8)
    plt.show()

# %% [markdown]
# # CMPSCI 389 HW2
# ##### Assigned: Feb 22 2024; Due: Feb 29 2024 @ 11:59 pm EST
# 
# It’s time! We’re making a proper Multi-layer neural network – well, you are at least.
# 
# Instead of just doing matrix multiplication we’re also going to include non-linearities. This can be a bit of a headache (not gonna lie to you), but if you think about the computation graph and have the patience to read my dumb comments you’ll be fine – I promise! And yes you still need to use Latex, I’m not backing down on that.

# %% [markdown]
# ### 1) Primer – what’s different?
# So, now that we’re in the realm of multi-layer perceptrons (MLP), things are going to get a little more complicated. In particular we are going to have multiple layers (essentially perceptrons with multiple outputs), which will have weight **matrices**, denoted as ($W_{0}$,$W_{1}$,...,$W_{L}$ for $L$ layers) which will have dimensions of (input size, output size), where the input and output size are the corresponding input and output size of that layer – size is the number of neurons in that layer.
# 
# For example we could have a two layer MLP which would then have a $W_{0}$ and $W_{1}$. The data given may have $M$ input features and $C$ output features. In this case we know the sizes of the weight matrices must have be $W_{0} = (M, A)$ and $W_{1} = (A, C)$ where A is some number that we choose. The important part of this is that the first layer of weights, $W_{0}$, must have first dimension equal to the size of the input and the final layer of weights must have final dimension equal to the output size. Everything else is up to us – as long as they match. Hope that landed, but I bet it will by the end of the HW.
# 
# ![example neural network](L_Layer_Net.png)
# 
# \begin{align}
#     \text{Figure 1: Example of an L layer neural network – **this is not including a nonlinearity**}
# \end{align}
# 
# The second major change is that we will include a nonlinearity operation, *σ*, after every layer of our MLP, **besides the final layer**. This *σ* must be a nonlinear function and will be taken on the entire output of each layer.
# 
# Because of these two changes we must make use of backpropogation (finding the gradient layer by layer without repeat- ing calculation) and must also remember the output of each layer of the NN (you’ll see why when we do the computation graph – yeah I’m going to make you do one of those)
# 
# In the coding part we’re going to refer to the output of a certain layer, *layer*, and its corresponding nonlinearity as *$A_{layer+1}$* . This would make the output of the first ($W_{0}$) layer be represented by $A_{1}$. If you actually read all of this use the *$A_{layer+1}$* notation in the computation graph so I know you’re a ~~tryhard~~ good student.

# %% [markdown]
# ## 2) Math! and other stuff that will help you (30 points)
# Do the math:
# 
# 1. For the following nonlinearity, calculate the derivative of it w.r.t (with respect to) its input (this is kind of tricky, but if you get stuck the answers in lecture (you didn’t hear it from me). Give it a try first though, pretty please):
# 
# \begin{align}
#     σ(X) = \frac{1}{1+e^{-X}}
# \end{align}
# 
# \begin{align}
#     \frac{∂σ}{∂X}=\frac{e^{-X}}{(1+e^{-X})^2} = (1-sig(x))(sig(x))
# \end{align}

# %% [markdown]
# \begin{align}
#     \frac{∂σ}{∂X}=\frac{e^{-X}}{(1+e^{-X})^2} = (1-sig(x))(sig(x))
# \end{align}

# %% [markdown]
# 2. Imagining that we’re using this nonlinearity after a single layer of neural network – calculate the derivative of the output w.r.t the weights.
# 
# \begin{align}
#     \hat{y} = X \cdot W + b \text{  and  } σ(\hat{y}) = \frac{1}{1+e^{- \hat{y}}}
# \end{align}
# 
# also, recall:
# 
# \begin{align}
#     \frac{∂A}{∂C} = \frac{∂A}{∂B} * \frac{∂B}{∂C}
# \end{align}
# 
# solve: 
# \begin{align}
#     \frac{∂σ}{∂W}= 
# \end{align}

# %% [markdown]
# ### Check this with someone
# 
# \begin{align}
#     \frac{∂σ}{∂W}= X^T\cdot(\frac{1}{1+e^{-(X\cdot W + B)}})(1-\frac{1}{1+e^{-(X\cdot W + B)}})
# \end{align}

# %% [markdown]
# 3. Now something I’m sure you’ll love – draw a computation graph of a 2 layer neural network (2 weight matrices – using dot product as a basic operation) with nonlinearity, σ, between the layers. I.E :
# 
# \begin{align}
#     \hat{y} = (σ(X\cdot W_{0}+b_{0})) \cdot W_{1} + b_{1}
# \end{align}

# %% [markdown]
# ![Computational Graph](img1.jpg)

# %% [markdown]
# 4. Now time to use the computation graph! Use it to solve the following (Show the dimensions match too!):
# 
# \begin{align}
#     \frac{∂\hat{y}}{∂b_{0}} = W_1 \frac{e^{-x}}{(1-e^{-x})^2} = \frac{d\hat{y}}{d\hat{y}}\cdot W_1^T sig(x) (1-sig(x))
# \end{align}
# 
# \begin{align}
#     \frac{∂\hat{y}}{∂W_{0}} = W_1 \frac{e^{-x}}{(1-e^{-x})^2}X = X^T \cdot (\frac{d\hat{y}}{d\hat{y}}\cdot W_1^T sig(x) (1-sig(x)))
# \end{align}

# %% [markdown]
# ### 3) CODING! (60 points)
# So I’ll level with you – this is gonna be a little bit harder than the last one. But using everything you know you’ll be able to do it cause I believe in you! BTW **you need to pip install a package called tqdm** for this hw
# 
# Primarily, we have a somewhat strange (at first at least) choice to use a dictionary to store everything. This will become tremendously useful though when it comes to doing backpropogation so just trust me I guess ¯\\_(ツ)_/¯. You can just run the first block, then beyond that you'll need to fill out each TODO.

# %%
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from tqdm import tqdm

def load_data(file_path: str, label: str)->Tuple[np.ndarray, np.ndarray]:
    '''
    This function loads and parses text file separated by a ',' character and
    returns a data set as two arrays, an array of features, and an array of labels.
    Parameters
    ----------
    file_path : str
                path to the file containing the data set
    label : str
                A label of whether to grab training or testing data
    Returns
    -------
    features : ndarray
                2D array of shape (n,c) containing features for the data set
    labels : ndarray
                2D array of shape (n,d) containing labels for the data set
    '''
    D = np.genfromtxt(file_path, delimiter=",")
    if label == "train":
        features = D[0:800, :-3]  # all columns but the last three
        labels = D[0:800, -3:]  # the last 3 columns 
    else:
        features = D[801:1000, :-3]  # all columns but the last three
        labels = D[801:1000, -3:]  # the last 3 columns
    
    return features, labels

# %% [markdown]
# 1. **Initialize the network:** complete the *initialize_network()* method. 
# 
#     For this you’ll recieve a list of layer sizes, *layer_sizes*, which will denote the layer sizes with the first being the input size and the last being the output size, this will tell us what sized weight matrices we'll need.
# 
#     **Example:** layer_size = [100,1], this will denote a 1-layer network (like a Perceptron), with inputs with 100 features and 1 feature output. Here we'd want a (100 x 1) weight matrix.
# 
#     **Example 2:** layer_size = [100, 20, 1], here we'd want a 2-layer network that took the same size input and generated the same size output, but first multiplies to size 20 first. here we would need to weight matrices (one of size: (100,20) and another of size (20,1))  
# 
#     For the weight matrices and bias terms initialize using *np.random.randn()* to sample random numbers between [-1, 1] from a normal distribution (or bell curve or gaussian, all the same). Then use the scale value to scale all of them. These need to be stored in the given dictionary such that the first weight matrix is stored as ”W0” and first bias is ”b0”.

# %%
def initialize_network(layer_sizes : list, scale : float):
    """
    This function will inialize weights for an arbitrary neural network defined by
    the number of neurons in each layer, given by 'layer_sizes'
    Your weights should be initialized to be numpy arrays composed of random numbers
    taken from a gaussian distribution with mean=0 and std=1, then multiplied by scale
    Parameters
    ----------
    layer_sizes : np.ndarray
        An array of intergers that denote the number of neurons in each layer
        [100, 50, 20] would denote a network with 100 inputs, a hidden layer
        with 50 neurons and output of size 20 -- this would mean W_0 would have dimensions (100,50) for example
    scale : float
        The scale of our initial weights, our weights should mostly be in the range
        [-1,1] * scale
    
    Returns
    ---------
    init_params : dict
        A dictionary that maps labels for parameters to an array of those parameters' 
        initial values
        You MUST have 'W0' map to the first weight matrix, 'W1' to the second, etc. 
        Hint: "W" + str(1) is "W1"
        AND have the first biases similarly be "b0", "b1", etc
    """

    init_params = {}

    # TODO Initialize the parameter dictionary with weight matrices and biases with random values
    # You need to use np.random.randn() to do this -- you can look up the API
    # This will give a number sampled from a normal distribution (a bell curve) 
    for i in range(1, len(layer_sizes)):
        init_params['W' + str(i - 1)] = np.random.randn(layer_sizes[i - 1], layer_sizes[i]) * scale
        init_params['b' + str(i - 1)] = np.random.randn(layer_sizes[i]) * scale

    return init_params

# %%
exmaple_layers = [100, 50, 20, 5]
example_scale = 1
your_output = initialize_network(exmaple_layers, example_scale)

print("your output type:",type(your_output) )
print("your output length:", len(your_output))
print("You should have a dictionary with 6 terms")

print(your_output['W2'])
print(your_output['W2'].shape)
print("This should be a 20 x 5 numpy matrix with very small values")
# print([your_output[i].shape for i in your_output])

# %% [markdown]
# 2. Next you need to complete the forward for the nonlinearity. Complete the *sigma_forward()* method. The forward is just the calculation of the output of the nonlinear function given some input, IN.

# %%
def sigmoid_forward(OUT: np.ndarray):
    """
    performs the nonlinear function (sigmoid) on the given input and returns the result
    this is 1/(1 + e^-OUT)

    Parameters
    ----------
    OUT: np.ndarray 
        The given output of a layer's matrix multiplication

    Returns
    ----------
    A: np.ndarray 
        sigma(OUT), this is the output of your sigmoid function given input, OUT,
        'A' and 'OUT' are used because these are the labels we will use in the next step,
        for now you can ignore the naming.
    """

    ######################################

    # TODO Implement the sigmoid function 
    A = 1 / (1 + np.exp(-OUT))
    ######################################

    return A

# %%
example_inputs = np.arange(-5,5,0.25)
your_output = sigmoid_forward(example_inputs)

plt.plot(example_inputs, your_output)
plt.title("Your sigmoid function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()

# %% [markdown]
# 3. After this you can put it together with your model to make your full *forward()* method, which will do a forward pass through your entire model (each layer, including the nonlinearities). It’s important to note that the final layer is not followed by a nonlinearity. You also must save every intermittent value in the cache (again for use later).
# 
# It is important to not only do the forward pass, but also cache the intermittent values so that we (well, *you* ) can do backpropogation next.

# %%
def forward(params: dict, X: np.ndarray):
    """
    This function will perform the forward pass of your backprop algorithm
    It will calculate your networks prediction using your parameters and
    will keep a cache of all intermittent values (which you need for backprop)
    ** YOU MUST COMPLETE THE "sigmoid_forward()" method for this part **
    Parameters
    ----------
    params : dict
        A dictionary that maps the labels: 'W0', 'W1', etc to their respective
        weight matrices -- this is the current state of your params
    X : np.ndarray
        A 2D numpy array representing input where each row represents a feature vector
    
    Returns
    ---------
    prediction : np.ndarray
        A 1D numpy array holding the predictions of your network given input X
    cache : dict
        A dictionary that holds all of the intermittent values calculated during your
        forward pass of your network (the 'OUT' and 'A'  of each layer), you must have the
        keys of this dictionary be of the form "AL" and "OUTL" where "AL" representes the input
        to the L-th layer of weights and "OUT(L+1)" is the output after multiplying by weights in Layer L . 
        
        X = OUT0 = A0
        XW0 + b0 = OUT1
        sigmoid(XW0 + b0) = A1
        sigmoid(XW0 + b0)W1 + b1 = OUT2
        ... 

        i.e "OUT0" will be the key for exactly the input X and "A0" will be (as a special case) also X  
        generally "AL" will be sigma("OUTL")
    """
    cache = {}

    cache["A0"] = np.copy(X)

    # TODO -- implement the forward pass of your network
    A = X
    # print(f'This is X - {cache["A0"]}')
    for l in range(1, int(len(params) / 2)):
        
        A_i_minus_1 = A
        out_i = np.dot(A_i_minus_1, params['W' + str(l-1)]) + params['b' + str(l-1)]
        cache['OUT' + str(l)] = out_i
        A = sigmoid_forward(out_i)
        cache['A' + str(l)] = A

    try: out_last = np.dot(A, params['W' + str(int((len(params) / 2)) - 1)]) + params['b' + str(int((len(params) / 2)) - 1)]
    except: print('**********************************', params, f'X : {X.shape}')
    cache['OUT' + str(int(len(params) / 2))] = out_last
    prediction = out_last  

    return prediction, cache

# %% [markdown]
# Use this code to check that your forward pass works:

# %%
x = np.random.rand(3, 5)

print()
print("Single layer Network ----------------------------")
print()

params = {"W0": np.ones((5,4)), "b0": np.zeros((4))}

your_pred, _ = forward(params, x)

print("your prediction:", your_pred[1:2])
print("correct prediction:", (x.dot(params["W0"]) + params["b0"])[1:2])

print()
print("Now larger Network ----------------------------")
print()

params["W1"] = np.ones((4,2))
params["b1"] = np.zeros((2))

your_pred, your_cache = forward(params, x)

print("your ouput size with larger net:", your_pred.shape, "-- Should be (3,2)" )
print("your Cache (check this looks right):")
for key in your_cache:
    print()
    print(key,":", your_cache[key])
    print("Shape of ", key, ":", your_cache[key].shape)

# %% [markdown]
# 4. Next we are going to start doing the hard part, derivatives! In this step though all you need to do is complete *sigma_backward()* which will calculate the derivative of sigma with respect to its input (whoa! like the math we did!). This method takes its output as an input – why on earth would it do that?? Check out your math! and if it still doesn’t make sense you should check your math.

# %%
def sigmoid_backward(A: np.ndarray):
    """
    calculates the derivative of your sigma function give the output of it
    Parameters
    ----------
    A: np.ndarray 
        sigmoid(OUT), this is the A value (output of the sigma) of 
        some layer. This is all we need to find dsigma / dOUT believe it or not
    Returns
    ----------
    dsigmoid: np.ndarray
        the derivative of sigmoid(OUT) dOUT -- this will use the A value
        it will also be very simple
    """

    ######################################

    # TODO Implement the derivative of sigmoid 

    dsigmoid = A * (1 - A)

    return dsigmoid

# %%
example_A = [1, 0.75, 0.5, 0.25, 0]
your_dsigmoid = sigmoid_backward(np.array(example_A))

print("yours:",your_dsigmoid)
print("correct:", [0, 0.1875, 0.25, 0.1875, 0])

example_inputs = np.arange(0,1,0.05)
your_output = sigmoid_backward(example_inputs)

plt.plot(example_inputs, your_output)
plt.title("Your sigmoid_backward function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()


# %% [markdown]
# 5. Well it’s later, time to use the cache! This next step is the heart of your entire homework so it may take a little bit. You are going to complete the *backprop_and_loss()* method which will take your network, your cache, and the true output and calculate the loss as well as the gradient of loss with respect to every parameter (i.e. each ”WL” and ”bL”) and store those gradients in a cache using the same naming convention. **Note:** this uses MEAN squared error
# 
# **HINT:** You'll need to loop over the layers (backwards) to calculate the intermediate derivatives and use them to calculate the next layer's derivatives. If you are able to do a single layer fully, you should be able to iteratively do it. 

# %%
def backprop_and_loss(params: dict, prediction: np.ndarray, cache: dict, Y : np.ndarray):
    """
    This function will calculate the loss (LSE) of your predictions and the gradient
    of your network for a single iteration. To calculate the gradient you must
    use backpropogation through your network
    ** YOU MUST COMPLETE THE "sigmoid_backward()" method for this part **
    Parameters
    ----------
    params : dict
        A dictionary that maps the labels: 'W0', 'W1', etc to their respective
        weight as well as 'b0', 'b1', etc to the bias
        -- this is the current state of your params
    prediction : np.ndarray
        A 1D numpy array holding the predictions of your network given input X
    cache : dict
        A dictionary that holds all of the intermittent values calculated during your
        forward pass of your network (the 'OUT' and 'A'  of each layer), you must have the
        keys of this dictionary be of the form "AL" and "OUTL" where "AL" representes the input
        to the L-th layer of weights and "OUT(L+1)" is the output after multiplying by weights in Layer L . 
        
        i.e "OUT0" will be the key for exactly the input X and "A0" will be (as a special case) also X 
        generally "AL" will be sigma("OUTL")
    Y : np.ndarray
        A 1D numpy array of the correct labels of our input X
    Returns
    ---------
    gradient : dict
        A dictionary that maps the labels: 'W0', 'W1', etc to the gradients of Loss with respect to 
        the respective parameters (eg 'W0' -> gradient of Loss with respect to the first weight matrix)
    loss : float
        The MEAN (use np.mean) Squared Error loss given our predictions and true labels 'Y'. 
    
    """

    gradient = {}
    layers = int(len(params) / 2)
    loss = np.mean((prediction - Y) ** 2)
    dL_dAiPlus1 = 2 * (prediction - Y)
    dAL_dAi = dL_dAiPlus1
    # else: dAL_dAi = dL_dAiPlus1 * sigmoid_backward(cache['A' + str(layers)])
    Ai_minus1 = cache['A' + str(layers-1)]
    Wi = params['W' + str(layers-1)]
    dAL_dAi = dL_dAiPlus1
    gradient['W' + str(layers-1)] = np.dot(cache['A' + str(layers-1)].T, dAL_dAi)
    gradient['b' + str(layers-1)] = np.sum(dAL_dAi, axis=0, keepdims=True)
    dL_dAiPlus1 = np.dot(dAL_dAi, Wi.T)
    
    for l in range(layers - 1, 0, -1):
        Ai_minus1 = cache['A' + str(l-1)]
        Wi = params['W' + str(l-1)]
        dAL_dAi = dL_dAiPlus1 * sigmoid_backward(cache['A' + str(l)])
        if layers > 1: dL_dAiPlus1 = np.dot(dAL_dAi, Wi.T)
        gradient['W' + str(l-1)] = cache['A' + str(l-1)].T.dot(dAL_dAi)
        gradient['b' + str(l-1)] = np.sum(dAL_dAi, axis=0, keepdims=True)

    return gradient, loss
    

# %% [markdown]
# Use this code to test your backprop and loss function, this test is not extensive though, you should try more network sizes using your math

# %%
x = np.random.rand(3, 5)

print()
print("Single layer Network ----------------------------")
print()

params = {"W0": np.random.rand(5,1), "b0": np.random.rand(1)}

your_pred, your_cache = forward(params, x)
y = np.random.rand(3,1) 

your_grad, your_loss = backprop_and_loss(params, your_pred, your_cache, y)

print("your grad for W0:", your_grad["W0"])
print("correct grad for W0:", x.T.dot(2*(your_pred-y)))

print("your loss:", your_loss)
print("correct loss:", np.mean((your_pred-y)**2) )

# %% [markdown]
# 6. Now it’s time to actually do the learning! You must complete the *gradient_descent()* method which will call all of your other methods to find first the prediction, then the loss and gradient. It will then use them to do gradient descent on the entire parameter dictionary (using the corresponding gradients). You’re free to do this however you see fit (as long as it works!)

# %%
def gradient_descent(X : np.ndarray, Y : np.ndarray, initial_params : dict, lr : float, num_iterations : int)->Tuple[List[float], np.ndarray]:
    """
    This function runs gradient descent for a fixed number of iterations on the
    mean squared loss for a linear model parameterized by the weight vector w.
    This function returns a list of the losses for each iteration of gradient
    descent along with the final weight vector.
    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array representing input where each row represents a feature vector
    Y : np.ndarray
        A 1D numpy array where each element represents a label for MSE
    initial_params : dictionary
        A dictionary holding the initialization of all parameters of the model as np.ndarrays
        (e.g. key 'W0' maps to the first weight array of the neural net) 
    lr : float
        The step-size parameter to use with gradient descent.
    num_iterations : int
        The number of iterations of gradient descent to run.
    Returns
    -------
    losses : list
        A list of floats representing the loss from each iteration and the
        loss of the final weight vector
    final_params : dictionary 
        A dictionary holding all of the parameters after training as np.ndarrays
        (this should have the same mapping as initial_params, just with updated arrays) 
    """

    losses = []
    final_params = initial_params

    for n in tqdm(range(num_iterations)):  #tqdm will create a loading bar for your loop
        pred_and_cache = forward(final_params, X)
        grad_andloss = backprop_and_loss(final_params, pred_and_cache[0], pred_and_cache[1], Y)
        for key in final_params.keys():
            expected_shape = final_params[key].shape
            grad_andloss[0][key] = grad_andloss[0][key].reshape(expected_shape)
            final_params[key] -= lr * grad_andloss[0][key]
        losses.append(grad_andloss[1])
    
    return losses, final_params

# %% [markdown]
# 7. Now you can run it! And once you fix the 1000 bugs that you have... you can run it again and hopefully you see something *somewhat* like below – if you do, congrats! You (probably) have it working:
# 
# <img style="display: block; margin: auto;"
# src="ex_training.png">
# <!-- ![alt text](ex_training.png#center) -->
# 
# 
# \begin{align}
#     \text{Figure 2: Example Learning curve plot with default hyperparameters.}
# \end{align}
# 
# Run both of these blocks and see if your graph looks similar

# %%
def learning_curve(losses: list, names : list):
    """
    This function plots the learning curves for all gradient descent procedures in this homework.
    The plot is saved in the file learning_curve.png. No TODO here
    Parameters
    ----------
    losses : list
        A list of arrays representing the losses for the gradient at each iteration for each run of gradient descent
    names : list
        A list of strings representing the names for each gradient descent method
    Returns
    -------
    nothing
    """
    for loss in losses:
        plt.plot(loss)
    plt.xscale("log")
    plt.ylim(0, 10000)
    plt.xlabel("Iterations")
    plt.ylabel("Squared Loss")
    plt.title("Gradient Descent")
    plt.legend(names)
    plt.savefig("learning_curve.png")
    plt.show()


# %%
global training_losses
train_losses = []

def train_networks():
    Train_X, Train_Y = load_data("StudentsPerformance.csv", "train")  # load the data set

    N = 1000 # N needs to equal 10,000 for your final plot. You can lower it to tune hyperparameters.

    init_params0 = initialize_network([17,3], scale=0.1) # initializes a sigle layer network (perceptron)
    losses0, final_params0 = gradient_descent(Train_X, Train_Y, init_params0, lr=1e-6, num_iterations=N)  

    init_params1 = initialize_network([17, 5, 3], scale=0.1)  # initializes a two layer network
    losses1, final_params1 = gradient_descent(Train_X, Train_Y, init_params1, lr=1e-6, num_iterations=N)  

    init_params2 = initialize_network([17, 7, 3, 3], scale=0.1)  # initializes a many layer network
    losses2, final_params2 = gradient_descent(Train_X, Train_Y, init_params2, lr=1e-6, num_iterations=N)   

    all_losses = [losses0, losses1, losses2]
    names = ["single layer", "two layer", "many layer"]
    print("final training loss values")
    for name, losses in zip(names, all_losses):
        print("{0:.<21}{1:>8.1f}".format(name, float(losses[-1])))

    learning_curve(all_losses, names)

    # TESTING 

    Test_X, Test_Y = load_data("StudentsPerformance.csv", "test")

    pred0, _ = forward(final_params0, Test_X)
    test_loss0 = np.sum(np.square(Test_Y  - pred0)) 
    print("test loss of model 1:", test_loss0)

    pred1, _ = forward(final_params1, Test_X)
    test_loss1 = np.sum(np.square(Test_Y  - pred1)) 
    print("test loss of model 2:", test_loss1)

    pred2, _ = forward(final_params2, Test_X)
    test_loss2 = np.sum(np.square(Test_Y  - pred2)) 
    print("test loss of model 3:", test_loss2)

    # TODO choose the hyper parameters for your best model (change them in train_best_model() )
    # You'll have to uncomment the below lines for once you find your best model

    best_losses, best_params = train_best_model(Train_X, Train_Y) 
    best_pred, _ = forward(best_params, Test_X)
    best_loss = np.sum(np.square(Test_Y - best_pred)) 
    print("test loss of your \"best\" model:", best_loss)
    return [losses0, losses1, losses2], [test_loss0, test_loss1, test_loss2, best_loss]


all_3_training_losses, all_4_testing_losses = train_networks()

# %% [markdown]
# ### 4a) That other bit you forgot about until now (10 points)
# 1. Use *train_best_model()* (you’ll need to find some good hyperparameters) and plot its training alongside the other 3 models (you’ll need to change code in main to plot this) and show it here – Describe your best model and list the values of its hyperparameters here. You should be able to do significantly better than the default, you will be checked for the code you used to find a best model and for the quality of your final loss.

# %%
def train_best_model(Train_X, Train_Y):
    """
    This function will train the model with the hyper parameters
    and layers that you have found to be best -- this model must get below 3
    MSE loss on our test data (which is not the test data you are given)
    """

    # TODO CHANGE THESE VALUES -- you must experiment to find good values
    # Maybe you can use a loop to try many!


    lowest_loss = float('inf')

    for s in [0.01, 0.05, 0.1, 0.5, 1]:
        for layers in [[17,3], [17,5,3], [17,7,3,3]]:
            for lr in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]:
                for iters in [100, 200, 400, 800, 1600, 3200]:
                    curr_pars = initialize_network(layers, s)
                    loss_arr, pars = gradient_descent(Train_X, Train_Y, curr_pars, lr, iters)
                    if loss_arr[-1] < lowest_loss:
                        lowest_loss = loss_arr[-1]
                        best_losses = loss_arr
                        best_pars = pars
                        BEST_SCALE, BEST_LAYERS, BEST_ALPHA, BEST_NUM_ITERATIONS = s, layers, lr, iters

    print(f'BEST_SCALE = {s}, BEST_LAYERS = {layers}, BEST_ALPHA = {lr}, BEST_NUM_ITERATIONS = {iters}')
    return best_losses, best_pars
                    

    best_params = initialize_network(BEST_LAYERS, BEST_SCALE)
    best_losses, best_final_params = gradient_descent(Train_X, Train_Y, best_params, lr=BEST_ALPHA, num_iterations=BEST_NUM_ITERATIONS)

    return best_losses, best_final_params

Train_X, Train_Y = load_data("StudentsPerformance.csv", "train")  # load the data set  
best_losses,best_final_params = train_best_model(Train_X, Train_Y)

print("final loss:", best_losses[-1])   
all_3_training_losses.append(best_losses)
all_4_training_losses = all_3_training_losses #Basically just renaming "all_3_training_losses" to "all_4_training_losses" because I appended an element

# %% [markdown]
# The best model is the one with the lowest curve as you can see in the graph. The value of its hyper-parameters have been printed above. 

# %% [markdown]
# 2. Now note the printed test losses (the loss on the test set which isn’t trained on), for the 3 original models and your best, show the train loss and test loss. What is the relationship between these, explain why you think this is?

# %% [markdown]
# The values of final training losses and testing losses have been printed and shown below. As we can see from the graph above, the training curve of our best model ends lowest in a significant fashion. This is clearly because we performed hyper-parameter tuning on our last model, due to which we were able to identify the specific layers, scale, learning-rate, and number of iterations at which our model performs the best. We tested our model on each of these several permuations and combinations of these hyper-parameters (as can be seen in the code above), and chose those hyper-parameters where the losses were the lowest. Which is why our last model performed significantly better than the other 3. 

# %%
names = ["single layer", "two layer", "many layer", "best"]
learning_curve(all_4_training_losses, ["single layer", "two layer", "many layer", "best"])
print("final training loss values")
for name, losses in zip(names, all_4_training_losses):
    print("{0:.<21}{1:>8.1f}".format(name, float(losses[-1])))

print("The testing loss values are shown below - ")
for name, loss in zip(names, all_4_testing_losses):
    print("{0:.<21}{1:>8.1f}".format(name, float(loss)))

print(f'Best training loss = {all_4_training_losses[3][-1]} , Best testing loss = {all_4_testing_losses[3]}')



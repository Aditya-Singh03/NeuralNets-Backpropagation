# Multi-Layer Neural Networks with Backpropagation from Scratch

This project embarks on a hands-on journey into the heart of deep learning, where you'll construct a Multi-Layer Perceptron (MLP) neural network from the ground up. Through meticulous implementation and experimentation, you'll gain profound insights into the mechanics of backpropagation, the algorithm that fuels the learning process in neural networks.

## Project Overview

* **Core Objective:** Demystify the backpropagation algorithm by building a functional MLP neural network and training it on real-world data.
* **Key Concepts:**
    * Multi-layer perceptrons
    * Forward and backward propagation
    * Gradient descent optimization
    * Loss functions
    * Hyperparameter tuning
    * Overfitting and regularization

## Project Structure

* **Code:** `NeuralNetsAndBackprop.ipynb` encapsulates the Python implementation of the MLP, backpropagation, and training procedures.
* **Data:** `StudentsPerformance.csv` provides the dataset for training and evaluating the models.
* **Output:** `learning_curve.png` visualizes the training loss for various network architectures.
* **README:** This enhanced `readme.md` furnishes comprehensive project documentation.

## Getting Started

1. **Prerequisites:** Ensure you have the following libraries installed:

   ```bash
   pip install numpy matplotlib tqdm
   ```

2. **Run the Code:** Execute the project using:

   ```bash
   python NeuralNetsAndBackprop.py
   ```

## Implementation Highlights

* **Network Initialization:** Weights and biases are intelligently initialized using random values from a normal distribution, carefully scaled to promote stable training.
* **Forward Pass:** The forward propagation algorithm is meticulously implemented, performing matrix multiplications and applying the sigmoid activation function at each layer, culminating in predictions.
* **Backpropagation:** The core of the project lies in the implementation of backpropagation, meticulously calculating gradients for weights and biases using the chain rule and leveraging cached values from the forward pass.
* **Loss Function:** Employs Mean Squared Error (MSE) to quantify the discrepancy between predictions and true labels, guiding the learning process.
* **Gradient Descent:** The network's parameters are iteratively refined using gradient descent, gradually minimizing the loss function and enhancing prediction accuracy.
* **Learning Curves:** Training progress is dynamically visualized through learning curves, offering valuable insights into the convergence behavior of different models.
* **Hyperparameter Optimization:** A systematic exploration of hyperparameters is conducted to identify the optimal configuration, significantly improving the model's performance.

## Results and Insights

* **Training Dynamics:** The `learning_curve.png` plot showcases the evolution of training loss across various network architectures:
    * **Single-Layer:** A basic perceptron model.
    * **Two-Layer:** An MLP with a single hidden layer.
    * **Multi-Layer:** A deeper MLP with multiple hidden layers.
    * **Best Model:** The model with the lowest training loss after hyperparameter tuning.

* **Generalization Performance:** Testing loss on a held-out test set reveals the model's ability to generalize to unseen data.

* **Best Model Configuration:**

    * Architecture: [17, 5, 3] 
    * Scale: 0.1
    * Learning rate: 1e-06
    * Number of iterations: 3200

* **Overfitting Analysis:** The best model exhibits a notably lower training loss but a higher testing loss, indicating a degree of overfitting. The model has learned the training data intricacies too well, impacting its ability to generalize effectively.

## Future Directions

* **Regularization Techniques:** Implement L1 or L2 regularization to mitigate overfitting and improve generalization.
* **Advanced Optimization:** Explore alternative optimization algorithms like Adam or RMSprop to potentially accelerate convergence and enhance performance.
* **Activation Functions:** Experiment with diverse activation functions such as ReLU or tanh to observe their impact on training dynamics and model expressiveness.
* **Data Preprocessing:** Employ techniques like standardization or normalization to potentially improve training stability and facilitate convergence.

## Conclusion

This project offers a hands-on exploration of backpropagation and its pivotal role in training neural networks. By building an MLP from scratch, you've gained a deep appreciation for the intricate interplay of weights, biases, activations, and gradients that empower these powerful learning machines.

Feel free to delve into the well-commented code (`NeuralNetsAndBackprop.ipynb`) for a more in-depth understanding of the implementation.

If you have any questions or suggestions for further enhancements, don't hesitate to reach out!

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(Z):
    """
    Computes the sigmoid activation
    
        Args:
        Z: input values as a numpy value
        
        Returns:
        sigmoid function values
    """
    
    return 1 / (1 + np.exp(-Z))


def visualize_decision_boundary(X, Y, model, padding=.5, step_size=.01):
    """
    Visualizes a model's boundary in 2d feature space
    
        Args:
        X:         features (n_samples, n_features)
        Y:         true target (n_samples, 1)
        model:     a model's prediction function
        padding:   padding for the plotted space
        step_size: step-size for the prediction grid
        
        Returns:
        scatter plot of features showing true class labels
        and the model's decision boundary
    """
    # Set the feature space boundaries for plotting
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
        
    # Generate a point grid in feature space with step size
    # 'step_size' between points
    x1, x2 = np.meshgrid(np.arange(x_min, x_max, step_size), 
                         np.arange(y_min, y_max, step_size))
    
    # Use the model's 'predict' function to compute
    # prediction values for all grid points
    z = model(np.c_[x1.ravel(), x2.ravel()])
    z = z.reshape(x1.shape)
    
    # Plot a scatter and contours showing the decision 
    # boundary
    plt.contourf(x1, x2, z)    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Decision boundary')
    plt.scatter(X[:,0], X[:,1], c=Y, edgecolors='w')
    
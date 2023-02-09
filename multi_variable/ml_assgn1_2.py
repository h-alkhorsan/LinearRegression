from load_data_ex2 import *
from normalize_features import *
from gradient_descent import *
from calculate_hypothesis import *
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex2()

# Normalize
X_normalized, mean_vec, std_vec = normalize_features(X)

# After normalizing, we append a column of ones to X, as the bias term
column_of_ones = np.ones((X_normalized.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
X_normalized = np.append(column_of_ones, X_normalized, axis=1)

# initialise trainable parameters theta, set learning rate alpha and number of iterations
theta = np.zeros((3))
alpha = 0.0025
iterations = 100

# plot predictions for every iteration?
do_plot = True

# call the gradient descent function to obtain the trained parameters theta_final
theta_final = gradient_descent(X_normalized, y, theta, alpha, iterations, do_plot)
print(f"theta values: {theta_final}")
#########################################

x = np.array([(1650, 3), (3000,4)])
x_normalized, x_mean_vec, x_std_vec = normalize_features(x)

add_bias = np.ones((x_normalized.shape[0], 1))
x_normalized = np.append(add_bias, x_normalized, axis=1)

y1 = calculate_hypothesis(x_normalized, theta_final, 0)
y2 = calculate_hypothesis(x_normalized, theta_final, 1)

print(f"Prediction 1: {y1}")
print(f"Prediction 2: {y2}")

# Write your code here
# Create two new samples: (1650, 3) and (3000, 4)
# Calculate the hypothesis for each sample, using the trained parameters theta_final
# Make sure to apply the same preprocessing that was applied to the training data
# Print the predicted prices for the two samples

########################################/
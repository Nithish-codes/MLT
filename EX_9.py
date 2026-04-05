import numpy as np
import matplotlib.pyplot as plt
import csv

def load_data(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row['x']))
            y.append(float(row['y']))
    return np.array(x), np.array(y)

def kernel(point, xmat, tau):
    m, _ = np.shape(xmat)
    weights = np.asmatrix(np.eye(m))
    for j in range(m):
        diff = point - xmat[j]
        # .item() converts the 1x1 matrix result into a scalar number
        exponent = (diff * diff.T).item() / (-2.0 * tau**2)
        weights[j, j] = np.exp(exponent)
    return weights

def local_weight(point, xmat, ymat, tau):
    weights = kernel(point, xmat, tau)
    # Normal Equation: (X^T * W * X)^-1 * (X^T * W * Y)
    theta = (xmat.T * (weights * xmat)).I * (xmat.T * (weights * ymat.T))
    return (point * theta).item() # .item() ensures we return a scalar

def locally_weighted_regression(xmat, ymat, tau):
    m, n = np.shape(xmat)
    predictions = np.zeros(m)
    for i in range(m):
        predictions[i] = local_weight(xmat[i], xmat, ymat, tau)
    return predictions

# --- Execution ---
x, y = load_data('training_data_set_for_ex_9.csv')

# Prepare matrices using asmatrix
xmat = np.asmatrix(np.column_stack((np.ones(len(x)), x)))
ymat = np.asmatrix(y)

tau = 0.1
predictions = locally_weighted_regression(xmat, ymat, tau)

# Sort for plotting
idx = x.argsort()
x_sorted = x[idx]
predictions_sorted = predictions[idx]

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x_sorted, predictions_sorted, color='red', linewidth=2, label=f'LWR Fit (tau={tau})')
plt.title('Locally Weighted Regression (Clean Output)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

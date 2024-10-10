import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sympy as sp

# Constants
T = 5
delta_t = 0.01
t = np.arange(0, T + delta_t, delta_t)
N = t.size

# Load data
def load_data(file_path):
    """Load data from a text file."""
    return np.loadtxt(file_path)

# Discrete Fourier Transform
def dft(x):
    """Compute the Discrete Fourier Transform of the input signal."""
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        X[k] = sum(x[n] * np.exp(-2j * np.pi * k * n / N) for n in range(N))
    return X / N

# Find significant frequencies
def find_significant_frequencies(Y):
    """Identify significant frequencies from the DFT result."""
    delta_f = 1 / T
    magnitude = np.abs(Y)
    k_star, _ = find_peaks(magnitude[:N // 2])

    peaks = []

    for i in range(1, N - 1):
        if (magnitude[i] > magnitude[i - 1] and magnitude[i] > magnitude[i + 1] and abs(magnitude[i] - magnitude[i - 1]) > 1):
            peaks.append(i)

    frequencies = k_star * delta_f
    return frequencies

# Generate symbols for parameters
def generate_a_symbols(n):
    """Generate symbolic variables for parameters in SymPy."""
    return sp.symbols(f'a0:{n}')

# Calculate square error
def calculate_square_error(y, x, frequencies, a):
    """Calculate the square error between the model and the observed data."""
    n = len(a)
    expression = 0
    for i in range(len(y)):
        t = x[i]
        tmp = a[0] * t ** 3 + a[1] * t ** 2 + a[2] * t + a[n - 1] - y[i]
        for j in range(3, n - 1):
            of = 2 * np.pi * frequencies[j - 3] * t
            tmp += a[j] * sp.sin(of)
        expression += tmp ** 2
    return expression

# Least squares method
def custom_lsq(y, x, frequencies):
    """Fit a model to the data using the least squares method."""
    n = 4 + len(frequencies)
    a = generate_a_symbols(n)
    expression = calculate_square_error(y, x, frequencies, a)
    gradient = [sp.diff(expression, param) for param in a]
    solution = sp.solve(gradient, a)
    print("Found parameters:", solution)
    return solution


# Fitted model
def fitted_model(t, a, fund_frequencies):
    """Construct the fitted model using the found parameters."""
    n = len(a)
    model = a[0] * t**3 + a[1] * t**2 + a[2] * t + a[n - 1]
    for j in range(3, n - 1):
        model += a[j] * np.sin(2 * np.pi * fund_frequencies[j - 3] * t)
    return model

# Signal spectre visualization
def plot_signal_spectre(Y):
    """Plot the signal spectre for given transformation."""
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(N // 2) / T, np.abs(Y[:N // 2]), label='Signal spectre')
    plt.title('Signal spectre')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()

# Visualization functions
def plot_signal(t, signal, title, xlabel, ylabel, color='blue', label=None):
    """Plot a signal with specified properties."""
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, label=label, color=color, linestyle='-', marker='o', markersize=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if label:
        plt.legend()
    plt.grid()
    plt.show()

def plot_comparison(t, signal1, signal2, label1, label2):
    """Plot a comparison between two signals."""
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal1, label=label1, color='blue', linestyle='-', marker='o', markersize=2)
    plt.plot(t, signal2, label=label2, color='red', linestyle='-')
    plt.title('Comparison of Model and Input Data')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()

# Main execution flow
if __name__ == "__main__":
    # Load data
    y = load_data('f4.txt')

    # Compute DFT
    Y = dft(y)

    # Find significant frequencies
    significant_freq = find_significant_frequencies(Y)
    print("Significant frequencies:", significant_freq)

    # Plot signal spectre
    plot_signal_spectre(Y)

    # Call least squares method to find model parameters
    solution = custom_lsq(y, t, significant_freq)

    # Extract parameter values from the solution
    a_values = [solution[param] for param in solution]

    # Calculate square error
    square_error = calculate_square_error(y, t, significant_freq, a_values)
    print("Square error:", square_error)

    # Generate fitted values
    y_fitted = fitted_model(t, a_values, significant_freq)

    # Plot the input data and fitted model
    plot_signal(t, y, 'Input Data', 'Time', 'Amplitude')
    plot_signal(t, y_fitted, 'Fitted Model', 'Time', 'Amplitude', 'red')
    plot_comparison(t, y, y_fitted, 'Input Data', 'Fitted Model')

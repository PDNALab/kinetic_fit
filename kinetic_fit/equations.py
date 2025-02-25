import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

class ParameterFitter:
    def __init__(self, A_, t,  initial_guess=None):
        """
        Initializes the ParameterFitter.

        Parameters:
        - A_ (list of arrays): The observed data for each function.
        - t (array): The independent variable values (e.g., time points).
        - initial_guess (list): The initial guess for the shared parameters.
        """
        self.A_ = [np.array(a, dtype=np.float64) for a in A_]
        if len(self.A_) > 7:
            raise ValueError('This algorithm works only up to 6 modifications')
        self.n_mods = len(self.A_)
        self.t = np.array(t, dtype=np.float64)
        self.functions = [A_0, A_1, A_2, A_3, A_4, A_5, A_6][:self.n_mods]
        self.parms = None
        
        if initial_guess is None:
            self.initial_guess = np.array([1.0, 0.9, 0.6, 0.5, 0.4, 0.1, 0.0], dtype=np.float64)
            self.initial_guess[self.n_mods:] = [0.0] * (len(self.initial_guess) - self.n_mods)
            # self.initial_guess = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            self.initial_guess = np.array(initial_guess, dtype=np.float64)

    def fit(self):
        """
        Fits the shared parameters to the given data by minimizing the sum of squared residuals.
        """
        
        def objective_function(params):
            """Calculates the sum of squared residuals for all functions."""
            residuals = []
            res_important = [] # summ of all A should be 1 -- normalized
            for A, func in zip(self.A_, self.functions):
                params[self.n_mods:] = 0.0
                residuals.extend(A - func(self.t, *params))
            res_important.extend(np.ones_like(A, dtype=np.float64) - A_all(self.t, *params))
            return np.sum(np.array(residuals)**2) + np.sum(np.array(res_important)**2)

        result = minimize(objective_function, self.initial_guess, method='SLSQP', tol=1e-100,
                          bounds=[(0, 2)] * len(self.initial_guess))
        self.parms = result.x

    def get_parameters(self):
        """
        Returns the optimized parameters after fitting.
        """
        return self.parms
    
    def get_contour(self):
        """
        Returns the 2D array of (mod, time).
        """
        t_ = np.arange(0, np.max(self.t), 0.05, dtype=np.float64)
        return np.array([f(t_, *self.parms) for f in [A_0, A_1, A_2, A_3, A_4, A_5, A_6]], dtype=np.float64)


def A_i(t, *k_values):
    """
    Calculates A_i(t) for a given set of k values (k1, k2, ..., k_(i+1)).
    Designed for use with scipy.optimize.curve_fit.

    Args:
        t: Time variable.
        *k_values: Variable number of rate constants (k1, k2, ..., k_(i+1)).

    Returns:
        The value of A_i(t).
        Returns np.nan if any two k values are equal to avoid division by zero.
    """
    n = len(k_values)  # n=i in A_i

    if not 2 <= n:
        raise ValueError("The number of k_values must be higher than 2.")

    i = n - 1 

    product_k = np.prod(k_values[:i])
    sum_terms = 0

    for j in range(i + 1):
        numerator = np.exp(-k_values[j] * t)
        denominator = 1
        for p in range(i + 1):
            if p != j:
                denominator *= (k_values[p] - k_values[j])
        denominator = 1e-6 if abs(denominator) < 1e-6 else denominator # bypass dividing by zero
        sum_terms += numerator / denominator

    return product_k * sum_terms

# Define the functions to fit
def A_0(t, k1, k2, k3, k4, k5, k6, k7):
    return np.exp(-t * k1)

def A_1(t, k1, k2, k3, k4, k5, k6, k7):
    return A_i(t, k1, k2)

def A_2(t, k1, k2, k3, k4, k5, k6, k7):
    return A_i(t, k1, k2, k3)

def A_3(t, k1, k2, k3, k4, k5, k6, k7):
    return A_i(t, k1, k2, k3, k4)

def A_4(t, k1, k2, k3, k4, k5, k6, k7):
    return A_i(t, k1, k2, k3, k4, k5)

def A_5(t, k1, k2, k3, k4, k5, k6, k7):
    return A_i(t, k1, k2, k3, k4, k5, k6)

def A_6(t, k1, k2, k3, k4, k5, k6, k7):
    return A_i(t, k1, k2, k3, k4, k5, k6, k7)

def A_all(t, k1, k2, k3, k4, k5, k6, k7):
    return np.exp(-t * k1) + A_i(t, k1, k2) + A_i(t, k1, k2, k3) + A_i(t, k1, k2, k3, k4) + A_i(t, k1, k2, k3, k4, k5) + A_i(t, k1, k2, k3, k4, k5, k6) + A_i(t, k1, k2, k3, k4, k5, k6, k7)

def smooth_curve(x, y, degree=3):
    """
    Fit a smooth curve through the given (x, y) data points.

    Parameters:
    x (numpy.ndarray): 1D array of x-values
    y (numpy.ndarray): 1D array of y-values
    degree (int, optional): The degree of the polynomial to fit. Default is 3.

    Returns:
    numpy.ndarray, numpy.ndarray: Smoothed x-values, Smoothed y-values
    """
    # Fit a polynomial curve to the data
    coeffs = np.polyfit(x, y, deg=degree)
    smooth_curve_func = np.poly1d(coeffs)

    # Create a set of smooth x-values
    x_smooth = np.linspace(x.min(), x.max(), 100)

    # Calculate the corresponding y-values on the smooth curve
    y_smooth = smooth_curve_func(x_smooth)

    return x_smooth, y_smooth
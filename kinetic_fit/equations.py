import numpy as np
from scipy.optimize import minimize

class ParameterFitter:
    def __init__(self, A_, t,  initial_guess=None):
        """
        Initializes the ParameterFitter.

        Parameters:
        - A_ (list of arrays): The observed data for each function.
        - t (array): The independent variable values (e.g., time points).
        - initial_guess (list): The initial guess for the shared parameters.
        """
        self.A_ = A_
        if len(self.A_) > 6:
            raise ValueError('This algorithm works only up to 5 modifications')
        n_mods = len(self.A_)
        self.t = t
        self.functions = [A_0, A_1, A_2, A_3, A_4, A_5][:n_mods]
        self.parms = None
        
        if initial_guess is None:
            self.initial_guess = [1.2065, 0.7285, 0.5515, 0.9445, 1.0465, 0.0][:n_mods]
        else:
            self.initial_guess = initial_guess

    def fit(self):
        """
        Fits the shared parameters to the given data by minimizing the sum of squared residuals.
        """
        
        def objective_function(params):
            """Calculates the sum of squared residuals for all functions."""
            residuals = []
            for A, func in zip(self.A_, self.functions):
                # Set the last parameter to 0
                params[-1] = 0.0
                residuals.extend(A - func(self.t, *params))
            return np.sum(np.array(residuals)**2)

        result = minimize(objective_function, self.initial_guess)
        self.parms = result.x

    def get_parameters(self):
        """
        Returns the optimized parameters after fitting.
        
        Returns:
        - parms (array): The optimized parameters.
        """
        return self.parms

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
        # denominator = 1e-8 if abs(denominator) < 1e-8 else denominator # bypass dividing by zero
        sum_terms += numerator / denominator

    return product_k * sum_terms

# Define the functions to fit
def A_0(t, k1, k2, k3, k4, k5, k6):
    return np.exp(-t * k1)

def A_1(t, k1, k2, k3, k4, k5, k6):
    return A_i(t, k1, k2)

def A_2(t, k1, k2, k3, k4, k5, k6):
    return A_i(t, k1, k2, k3)

def A_3(t, k1, k2, k3, k4, k5, k6):
    return A_i(t, k1, k2, k3, k4)

def A_4(t, k1, k2, k3, k4, k5, k6):
    return A_i(t, k1, k2, k3, k4, k5)

def A_5(t, k1, k2, k3, k4, k5, k6):
    return A_i(t, k1, k2, k3, k4, k5, k6)

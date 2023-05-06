import time
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits import mplot3d
from math import exp
from math import log
from math import sqrt
from scipy.stats import norm
import mocaxpy

def call_option(spot, strike, time_to_maturity, volatility, risk_free_rate):
    nominator = log(spot / strike) + (risk_free_rate + 0.5 * (volatility * volatility)) * time_to_maturity
    denominator = volatility * sqrt(time_to_maturity)
    d1 = nominator / denominator
    d2 = d1 - denominator
    n_d1 = norm.cdf(d1)
    n_d2 = norm.cdf(d2)
    df = exp(-risk_free_rate * time_to_maturity)
    value = spot * n_d1 - strike * df * n_d2
    return value

# full revaluation
lowest_spot = 50
highest_spot = 150
n_spots = 1000
spots = np.linspace(lowest_spot, highest_spot, n_spots)

lowest_time = 0.25
highest_time = 2.5
n_times = 1000
times = np.linspace(lowest_time, highest_time, n_times)

full_revaluation_values = np.zeros(shape=(n_spots, n_times))
start_timer = time.perf_counter()

for i in range(n_spots):
    for j in range(n_times):
         full_revaluation_values[i, j] = call_option(spots[i], 100.0, times[j], 0.25, 0.01)

end_timer = time.perf_counter()
print('Full revaluation {} seconds.'.format(end_timer - start_timer))

# approximation by using MoCaX
n_dimension = 2
n_chebyshev_points = 25
domain = mocaxpy.MocaxDomain([[lowest_spot, highest_spot], [lowest_time, highest_time]])
accuracy = mocaxpy.MocaxNs([n_chebyshev_points, n_chebyshev_points])
mocax = mocaxpy.Mocax(None, n_dimension, domain, None, accuracy, None)
chebyshev_points = mocax.get_evaluation_points()
evaluated_values = np.zeros(len(chebyshev_points))

start_timer = time.perf_counter()
for i in range(len(chebyshev_points)):
    evaluated_values[i] = call_option(chebyshev_points[i][0], 100.0, chebyshev_points[i][1], 0.25, 0.01)

mocax.set_original_function_values(list(evaluated_values))
mocax_approximation_values = np.zeros(shape=(n_spots, n_times))

for i in range(n_spots):
    for j in range(n_times):
         mocax_approximation_values[i, j] = mocax.eval([spots[i], times[j]])

end_timer = time.perf_counter()
print('MoCax approximation {} seconds.'.format(end_timer - start_timer))

approximation_error = full_revaluation_values - mocax_approximation_values
print('Maximum approximation error {}.'.format(np.max(approximation_error)))

ax = pl.axes(projection='3d')
spot_grid, time_grid = np.meshgrid(spots, times)
ax.plot_surface(spot_grid, time_grid, approximation_error)
pl.show()

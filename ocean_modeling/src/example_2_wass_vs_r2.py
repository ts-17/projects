""" example_2_wass_vs_r2.py: to reproduce example 2. """
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

# generate synthetic data
np.random.seed(23583905)
n1, n2, n3 = 600, 100, 300
x1 = np.random.normal(-2, 0.5, size=n1) # wide gaussian on left
x2 = np.random.exponential(scale=1, size=n2) - 1 #skewed, heavy tail near -1
x3 = np.random.normal( 2, 0.5, size=n3) # tight gaussian on right
x = np.concatenate([x1, x2, x3])
np.random.shuffle(x)

# y = x^3 + scaler * noise
y = x**3 + 5 * np.random.randn(len(x))
X = x.reshape(-1, 1)

# model definitions
models = {
    'Linear Regression': LinearRegression(),
    'Polynomial (deg=3)': make_pipeline(PolynomialFeatures(degree=3), LinearRegression()),
    'KNN (k=20)': KNeighborsRegressor(n_neighbors=20),
}

# fit models and score
results = {}
for name, m in models.items():
    m.fit(X, y)
    y_pred = m.predict(X)
    results[name] = {
        'R2':        r2_score(y, y_pred),
        'W1 (Wasserstein)': wasserstein_distance(y, y_pred)
    }

# print results
print(f'{"Model":<25}{"R²":>8}{"W1":>15}')
for name, stats in results.items():
    print(f'{name:<25}{stats["R2"]:8.3f}{stats["W1 (Wasserstein)"]:15.3f}')

# plot
xx = np.linspace(x.min(), x.max(), 500)[:,None]
plt.scatter(x, y, s=10, alpha=0.4, label='Data')
for name, m in models.items():
    plt.plot(xx, m.predict(xx), label=name, lw=2)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('OLS vs Poly vs KNN on Heavy-Tail Mixture')
plt.show()

""" example_2_wass_vs_r2.py: to reproduce example 1. """
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

# generate synthetic data
np.random.seed(2362903)
clusters = [
    np.random.normal(-3, 0.3, size=100),
    np.random.normal(0, 0.3, size=800),
    np.random.normal(3, 0.3, size=100),
]
x = np.concatenate(clusters)
y = np.sin(x) + 0.1 * np.random.randn(len(x))   # underlying function sin(x) + noise

X = x.reshape(-1, 1)

# model definitions
models = {
    'Linear Regression': LinearRegression(),
    'Polynomial (deg=6)': make_pipeline(PolynomialFeatures(degree=6), LinearRegression()),
    'KNN (k=12)': KNeighborsRegressor(n_neighbors=12),
}

# fit models and score
results = {}
for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    results[name] = {
        'R2': r2_score(y, y_pred),
        'W1 (Wasserstein)': wasserstein_distance(y, y_pred)
    }

# print results
print(f'{"Model":<25}{"R²":>8}{"W1":>15}')
for name, stats in results.items():
    print(f'{name:<25}{stats["R2"]:8.3f}{stats["W1 (Wasserstein)"]:15.3f}')

# plot
plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=10, alpha=0.5, label='Data')
xx = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
for name, model in models.items():
    plt.plot(xx, model.predict(xx), label=name)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Model Fits')
plt.show()

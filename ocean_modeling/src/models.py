""" models.py defines model types, parameters, feature selection, etc. Several models 
were excluded from analysis due to compute time to train all of them adequately, so are
commented in this code.
"""

import numpy as np
import statsmodels.api as sm
from scipy.stats import wasserstein_distance
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
    HuberRegressor, SGDRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor
)
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from pygam import GAM
from config import RANDOM_STATE, PREDICTOR_COLS

# OLS wrapper for statsmodels
class OLSRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        Xc = sm.add_constant(X)
        self.model = sm.OLS(y, Xc).fit(disp=0) # filter warnings?
        return self
    
    def predict(self, X):
        Xc = sm.add_constant(X)
        return self.model.predict(Xc)

# GLM wrapper for statsmodels
class GLMRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        Xc = sm.add_constant(X)
        self.model = sm.GLM(y, Xc, family=sm.families.Gaussian()).fit(disp=0) # warnings?
        return self
    
    def predict(self, X):
        Xc = sm.add_constant(X)
        return self.model.predict(Xc)

# use SKL's wasserstein computation (lower = better)
def wasserstein_metric(y_true, y_pred):
    try:
        return wasserstein_distance(y_true, y_pred)
    except Exception as e:
        print(f"Error in Wasserstein metric: {e}")
        return np.nan

# forward feature selection
def forward_selection(X, y, model_class, param_grid=None, max_features=None, cv=5, scoring='r2'):
    n_features = X.shape[1]
    if max_features is None:
        max_features = n_features
    
    # start empty
    selected_features = []
    remaining_features = list(range(n_features))
    scores = []
    best_models = []
    
    # iterate until max is reached
    for i in range(max_features):
        best_score = -np.inf
        best_feature = None
        best_model = None
        
        # add an unused feature
        for feature in remaining_features:
            # store candidates
            candidate = selected_features + [feature]
            X_candidate = X[:, candidate]
            
            # evaluate model with candidate features
            if param_grid is not None and len(param_grid) > 0:
                try:
                    clean_param_grid = {}
                    for key, value in param_grid.items():
                        # remove prefixes (annoying) if present (ex. 'rdg__alpha' -->'alpha')
                        if '__' in key:
                            clean_key = key.split('__', 1)[1]
                            clean_param_grid[clean_key] = value
                        else:
                            clean_param_grid[key] = value
                    
                    grid = GridSearchCV(model_class(), clean_param_grid, cv=cv, scoring=scoring)
                    grid.fit(X_candidate, y)
                    score = grid.best_score_
                    model = grid.best_estimator_
                except Exception as e:
                    print(f"Grid search error for feature {feature}: {e}")
                    # fallback to basic model
                    model = model_class()
                    model.fit(X_candidate, y)
                    score = np.mean(cross_val_score(model, X_candidate, y, cv=cv, scoring=scoring))
            else:
                # if not grid searching
                model = model_class()
                try:
                    model.fit(X_candidate, y)
                    score = np.mean(cross_val_score(model, X_candidate, y, cv=cv, scoring=scoring))
                except Exception as e:
                    print(f"Error fitting model for feature {feature}: {e}")
                    score = -np.inf
            
            # if better, this is the new best
            if score > best_score:
                best_score = score
                best_feature = feature
                best_model = model
        
        # if not better, stop
        if best_feature is None:
            break
            
        # add best feature to selected set
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        scores.append(best_score)
        best_models.append(best_model)
        
        print(f"Step {i+1}: Added feature {PREDICTOR_COLS[best_feature]} (index {best_feature}), Score: {best_score:.4f}")
        
    return selected_features, scores, best_models


# model pipelines

def make_pipelines(cache_dir):
    # creates a dictionary of scikit-learn pipelines for various models.
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    poly3 = PolynomialFeatures(degree=3, include_bias=False)
    spline = SplineTransformer(degree=3, n_knots=5, include_bias=False)

    pipelines = {
        # statistical models
        'OLS': Pipeline([
            ('scl', StandardScaler()), 
            ('ols', OLSRegressor())
        ], memory=cache_dir),
        
        'GLM-2': Pipeline([
            ('p2', poly2), 
            ('scl', StandardScaler()), 
            ('glm', GLMRegressor())
        ], memory=cache_dir),
        
        'GLM-3': Pipeline([
            ('p3', poly3), 
            ('scl', StandardScaler()), 
            ('glm', GLMRegressor())
        ], memory=cache_dir),
        
        'Spline': Pipeline([
            ('spl', spline), 
            ('scl', StandardScaler()), 
            ('lr', LinearRegression())
        ], memory=cache_dir),
        
        # linear regularized models
        'Ridge': Pipeline([
            ('scl', StandardScaler()), 
            ('rdg', Ridge(solver='auto', random_state=RANDOM_STATE))
        ], memory=cache_dir),
        
        'Lasso': Pipeline([
            ('scl', StandardScaler()), 
            ('lso', Lasso(max_iter=10000, random_state=RANDOM_STATE))
        ], memory=cache_dir),
        
        'ElasticNet': Pipeline([
            ('scl', StandardScaler()), 
            ('en', ElasticNet(max_iter=10000, random_state=RANDOM_STATE))
        ], memory=cache_dir),
        
        # KNN
        'KNN': Pipeline([
            ('scl', StandardScaler()), 
            ('knn', KNeighborsRegressor())
        ], memory=cache_dir),
        
        # !!! note: several of the below are omitted to improve speed:
        # 'SGDRegressor': Pipeline([
        #     ('scl', StandardScaler()), 
        #     ('sgd', SGDRegressor(max_iter=1000, tol=1e-3, random_state=RANDOM_STATE))
        # ], memory=cache_dir),
        
        # Support Vector Machines
        # 'SVR-Linear': Pipeline([
        #     ('scl', StandardScaler()), 
        #     ('svr', SVR(kernel='linear', cache_size=500))
        # ], memory=cache_dir),
        
        # 'SVR-RBF': Pipeline([
        #     ('scl', StandardScaler()), 
        #     ('svr', SVR(kernel='rbf', gamma='scale', cache_size=500))
        # ], memory=cache_dir),
        
        # Neural Networks
        'NN-Small': Pipeline([
            ('scl', StandardScaler()), 
            ('mlp', MLPRegressor(hidden_layer_sizes=(20,), max_iter=1000, activation='relu', solver='sgd',
                                 early_stopping=True, warm_start=True, random_state=RANDOM_STATE))
        ], memory=cache_dir),

        'NN-Medium': Pipeline([
            ('scl', StandardScaler()), 
            ('mlp', MLPRegressor(hidden_layer_sizes=(50,20,20), max_iter=1000, activation='relu', solver='sgd',
                                 early_stopping=True, warm_start=True, random_state=RANDOM_STATE))
        ], memory=cache_dir),

        'NN-Large': Pipeline([
            ('scl', StandardScaler()),
            ('mlp', MLPRegressor(hidden_layer_sizes=(100,50, 50, 20,), max_iter=1000, activation='relu', solver='sgd',
                                 early_stopping=True, warm_start=True, random_state=RANDOM_STATE))
        ], memory=cache_dir),
        
        # # tree-based models
        # 'DecisionTree': Pipeline([
        #     ('scl', StandardScaler()), 
        #     ('dt', DecisionTreeRegressor(random_state=RANDOM_STATE))
        # ], memory=cache_dir),
        
        'RandomForest': Pipeline([
            ('scl', StandardScaler()), 
            ('rf', RandomForestRegressor(random_state=RANDOM_STATE))
        ], memory=cache_dir),
        
        # # boosting methods
        'GradientBoost': Pipeline([
            ('scl', StandardScaler()), 
            ('gb', GradientBoostingRegressor(random_state=RANDOM_STATE))
        ], memory=cache_dir),
        
        # 'HistGradBoost': Pipeline([
        #     ('scl', StandardScaler()), 
        #     ('hgb', HistGradientBoostingRegressor(random_state=RANDOM_STATE))
        # ], memory=cache_dir),
        
        'XGBoost': Pipeline([
            ('scl', StandardScaler()), 
            ('xgb', xgb.XGBRegressor(verbosity=0, random_state=RANDOM_STATE))
        ], memory=cache_dir),

        # GAM if time allows
    #   'GAM' = Pipeline([
    #         ('scl', StandardScaler()), 
    #         ('gam', GAM(distribution='normal', link='identity', terms='term'))
    #     ], memory=cache_dir)
    }
    
    return pipelines

# make a dictionary of param grids
def make_param_grids():
    param_grids = {
        'OLS': {}, 
        'GLM-2': {}, 
        'GLM-3': {}, 
        'Spline': {},
        'Ridge': {'rdg__alpha': [0.1, 1.0, 10.0]},
        'Lasso': {'lso__alpha': [0.01, 0.1, 1.0, 10.0]},
        'ElasticNet': {
            'en__alpha': [0.1, 1.0], 
            'en__l1_ratio': [0.1, 0.5, 0.9]
        },
        'KNN': {
            'knn__n_neighbors': [3, 5, 7, 11], 
            'knn__weights': ['distance'],
            'knn__algorithm': ['auto']
        },
        # 'SGDRegressor': {
        #     'sgd__alpha': [1e-4, 1e-3, 1e-2], 
        #     'sgd__loss': ['squared_error', 'huber', 'epsilon_insensitive']
        # },
        # Support Vector Machines
        # 'SVR-Linear': {
        #     'svr__C': [0.1, 1, 10], 
        #     'svr__epsilon': [0.1, 0.2]
        # },
        # 'SVR-RBF': {
        #     'svr__C': [0.1, 1, 10], 
        #     'svr__gamma': ['scale', 'auto', 0.1, 1.0]
        # },
        'NN-Small': {
            # 'mlp__alpha': [1e-4, 1e-2], 
            # 'mlp__activation': ['relu'],
            # 'mlp__solver': ['sgd'],
            'mlp__learning_rate_init': [0.001]
        },
        'NN-Medium': {
        #     'mlp__alpha': [1e-4, 1e-3, 1e-2], 
        #     'mlp__activation': ['relu'],
        #     'mlp__solver': ['adam'],
            'mlp__learning_rate_init': [0.001]
        },
        'NN-Large': {
        #     'mlp__alpha': [1e-4, 1e-3, 1e-2], 
        #     'mlp__activation': ['relu'],
        #     'mlp__solver': ['adam'],
            'mlp__learning_rate_init': [0.001]
        },
        # 'DecisionTree': {
        #     'dt__max_depth': [5, 10, None], 
        #     'dt__min_samples_split': [2, 5, 10]
        # },
        'RandomForest': {
            'rf__n_estimators': [10], 
            'rf__max_depth': [5], 
            'rf__min_samples_split': [5]
        },
        'GradientBoost': {
            'gb__n_estimators': [20, 50], 
            'gb__max_depth': [3, 6], 
            'gb__learning_rate': [0.01]
        },
        # 'HistGradBoost': {
        #     'hgb__max_iter': [100, 200], 
        #     'hgb__max_depth': [3, 6], 
        #     'hgb__learning_rate': [0.01, 0.1]
        # },
        'XGBoost': {
            'xgb__n_estimators': [20], 
            'xgb__max_depth': [3, 6], 
        #     'xgb__learning_rate': [0.01, 0.1], 
            'xgb__subsample': [0.8], 
            'xgb__colsample_bytree': [0.8]
        },
        #     'GAM': {
    #         'gam__n_splines': [10, 20], 
    #         'gam__lam': [0.1, 1.0, 10.0]
    #     }
    }
    return param_grids


# classes for feature selection
def get_model_classes():
    model_classes = {
        'OLS': OLSRegressor,
        'Ridge': Ridge,
        'Lasso': Lasso, 
        'ElasticNet': ElasticNet,
        'KNN': KNeighborsRegressor,
        'SVR': SVR,
        'DecisionTree': DecisionTreeRegressor,
        # 'RandomForest': RandomForestRegressor, # super slow?
        # 'ExtraTrees': ExtraTreesRegressor,
        # 'XGBoost': xgb.XGBRegressor # super slow?
        # 'GAM': GAM,
    }
    
    return model_classes
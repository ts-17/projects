""" main.py: execute the analysis of each model type and record results """
import warnings

from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')
from config import TEST_SIZE, RANDOM_STATE, RUN_FEATURE_SELECTION, PREDICTOR_COLS, RESPONSE_COL
from data import load_and_prep
from models import make_pipelines, make_param_grids, wasserstein_metric, forward_selection
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
    max_error,
    explained_variance_score,
    make_scorer
)
import numpy as np
import time
import json
import pandas as pd
import os
from datetime import datetime
from scipy.stats import wasserstein_distance

# negative wasserstein distance (since we need to maximize)
def neg_wasserstein_metric(y_true, y_pred):
    try:
        return -wasserstein_distance(y_true, y_pred)
    except Exception as e:
        print(f"Error in negative Wasserstein metric: {e}")
        return -np.inf

# custom scoring for param search
wasserstein_scorer = make_scorer(neg_wasserstein_metric)

def run():
    # timestamping outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"model_evaluation_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_and_prep()
    X, y = df[PREDICTOR_COLS].values, df[RESPONSE_COL].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    pipelines = make_pipelines(cache_dir='.')
    param_grids = make_param_grids()
    results = []
    
    for name, pipe in pipelines.items():
        print(f"\n>>> {name}")
        
        # optional forward selection
        selected_features = None
        if RUN_FEATURE_SELECTION and name in param_grids:
            selected_features, feature_scores, feature_models = forward_selection(
                Xtr, ytr,
                type(pipe.named_steps[list(pipe.named_steps)[-1]]),
                param_grids[name],
                max_features=len(PREDICTOR_COLS),
                cv=5,
                scoring=wasserstein_scorer  # wasserstein for feature selection too
            )
        
        # time training
        train_start_time = time.time()
        
        # originally used to slim the time for RF model, but this was omitted for time
        if name == 'RandomForest' or name == '':
            from sklearn.model_selection import RandomizedSearchCV
            gs = RandomizedSearchCV(
                pipe,
                param_grids[name],
                n_iter=5,
                cv=KFold(3, shuffle=True, random_state=RANDOM_STATE),
                scoring=wasserstein_scorer,
                n_jobs=-1
            )
        else:
            # for all others, GridSearchCV with Wasserstein scoring
            gs = GridSearchCV(
                pipe,
                param_grids[name],
                cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
                scoring=wasserstein_scorer,
                n_jobs=10,
                verbose=2
            )
            
        gs.fit(Xtr, ytr)
        train_time = time.time() - train_start_time
        
        # time inference
        inference_start_time = time.time()
        yp = gs.predict(Xte)
        inference_time = time.time() - inference_start_time
        
        # compute metrics
        r2 = r2_score(yte, yp)
        rmse = root_mean_squared_error(yte, yp)
        mae = mean_absolute_error(yte, yp)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.nanmean(np.abs((yte - yp) / yte)) * 100
        mx_err = max_error(yte, yp)
        expl_var = explained_variance_score(yte, yp)
        resid = yte - yp
        mean_resid = np.mean(resid)
        std_resid = np.std(resid)
        wass = wasserstein_metric(yte, yp)
        
        # store best params for this model
        best_params = gs.best_params_
        
        results.append({
            'model': name,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'max_error': mx_err,
            'explained_variance': expl_var,
            'mean_residual': mean_resid,
            'std_residual': std_resid,
            'wass': wass,
            'train_time': train_time,
            'inference_time': inference_time,
            'best_params': best_params,
            'selected_features': selected_features
        })
        
        # print timing results
        print(f"Training time: {train_time:.3f} seconds")
        print(f"Inference time: {inference_time:.3f} seconds")
        print(f"Wasserstein: {wass:.6f}")
        print(f"Best parameters: {best_params}")
    
    # dataframe from results for easy CSV export
    csv_results = []
    for r in results:
        csv_row = r.copy()
        # remove non-scalar values
        csv_row.pop('best_params', None)
        csv_row.pop('selected_features', None)
        csv_results.append(csv_row)
    
    # export to .csv file
    results_df = pd.DataFrame(csv_results)
    csv_path = os.path.join(output_dir, 'model_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # detailed report with best parameters
    detailed_report_path = os.path.join(output_dir, 'model_report.txt')
    with open(detailed_report_path, 'w') as f:
        # logging for organization
        f.write("===== MODEL REPORT =====\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test size: {TEST_SIZE}, Random state: {RANDOM_STATE}\n")
        f.write(f"Features: {', '.join(PREDICTOR_COLS)}\n")
        f.write(f"Response: {RESPONSE_COL}\n")
        f.write(f"Optimization metric: Wasserstein distance (lower is better)\n\n")
        
        # sorted for lower wass. is better:
        sorted_results = sorted(results, key=lambda x: x['wass'])
        
        for i, r in enumerate(sorted_results):
            f.write(f"MODEL #{i+1}: {r['model']}\n")
            f.write("="*50 + "\n")
            
            # metrics
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"  Wasserstein Distance: {r['wass']:.6f} (primary metric)\n")
            f.write(f"  R² Score: {r['r2']:.4f}\n")
            f.write(f"  RMSE: {r['rmse']:.4f}\n")
            f.write(f"  MAE: {r['mae']:.4f}\n")
            f.write(f"  MAPE: {r['mape']:.2f}%\n")
            f.write(f"  Max Error: {r['max_error']:.4f}\n")
            f.write(f"  Explained Variance: {r['explained_variance']:.4f}\n")
            f.write(f"  Mean Residual: {r['mean_residual']:.4f}\n")
            f.write(f"  Std Residual: {r['std_residual']:.4f}\n\n")
            
            f.write("TIMING METRICS:\n")
            f.write(f"  Training Time: {r['train_time']:.4f} seconds\n")
            f.write(f"  Inference Time: {r['inference_time']:.4f} seconds\n")
            f.write(f"  Wasserstein per Training Second: {r['wass_per_train_second']:.6f}\n")
            f.write(f"  Wasserstein per Inference Second: {r['wass_per_inference_second']:.6f}\n\n")
            
            # bests params
            f.write("BEST PARAMETERS:\n")
            for param, value in r['best_params'].items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
            
            # selected features (if feature selection was run)
            if r['selected_features'] is not None:
                f.write("SELECTED FEATURES:\n")
                for idx in r['selected_features']:
                    f.write(f"  {PREDICTOR_COLS[idx]} (index {idx})\n")
            f.write("\n\n")
    return results

if __name__ == '__main__':
    run()
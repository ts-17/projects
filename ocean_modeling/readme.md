# Repeat after me: I will not trust $R^2$.

This project investigates the limitations of $R^2$ as a regression evaluation metric and presents the Wasserstein distance as a more informative alternative for analyzing prediction quality. The analysis uses the California Cooperative Oceanic Fisheries Investigations (CalCOFI) dataset, which contains complex, high-dimensional oceanographic data.

This work demonstrates that $R^2$ can often fail to reflect meaningful structure in the residuals, particularly in cases of multimodal or heavy-tailed response distributions, leading to selection of non-optimal models. In contrast, the Wasserstein distance captures the full geometric discrepancy between predicted and observed distributions, including spatial misalignment and mass transport.

Models are evaluated using both conventional metrics ($R^2$, RMSE, MAE) and the 1-Wasserstein distance. Code includes synthetic benchmarks and real-world modeling pipelines using scikit-learn, with GridSearchCV for hyperparameter tuning and customized scoring functions.

## Contents
- `src/`: Code used in this project and analysis
- `evals/`: The for-score runs used in the paper
- `plots/`: Plots used in the paper
- `calcofi_data/`: Raw (and processed) CalCOFI dataset, and California coastline data from Natural Earth

## Note
Some models are commented out and use reduced hyperparameter grids to keep training time feasible. All work is performed on a Apple MacBook Pro (14 core M4 CPU, 48GB RAM).

## Reproducibility
All scripts set fixed random seeds. Runtime configurations and package versions are listed in `requirements.txt`. Results are generated using both synthetic and real datasets for replicability and comparison.
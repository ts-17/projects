import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib.lines import Line2D

data = pd.read_csv("model_evaluation_20250507_181859/model_results.csv")

scaler = MinMaxScaler()
data["train_time_scaled"] = scaler.fit_transform(data[["train_time"]])

def classify_family(model):
    if "GLM" in model:
        return "GLM"
    elif "NN-" in model:
        return "Neural Net"
    elif model in ["RandomForest", "GradientBoost", "XGBoost"]:
        return "Tree-Based"
    elif model in ["OLS", "Ridge", "Lasso", "ElasticNet"]:
        return "Linear"
    elif model == "KNN":
        return "KNN"
    elif model == "Spline":
        return "Spline"
    else:
        return "Other"

data["family"] = data["model"].apply(classify_family)

data = data.sort_values(by="wass").reset_index(drop=True)
family_list = sorted(data["family"].unique())
family_to_idx = {fam: i for i, fam in enumerate(family_list)}
data["family_idx"] = data["family"].map(family_to_idx)

cmap = plt.cm.plasma
norm = plt.Normalize(vmin=0, vmax=len(family_list) - 1)
data["color"] = data["family_idx"].apply(lambda i: cmap(norm(i)))

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(data["train_time_scaled"], data["wass"], s=60, c=data["color"])

# label the scatters
for i, row in data.iterrows():
    dx = 0.002
    dy = 0.002
    ax.text(row["train_time_scaled"] + dx, row["wass"] + dy, row["model"],
            fontsize=10, color="black")

#make a legend for model type
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=fam,
           markerfacecolor=cmap(norm(family_to_idx[fam])), markersize=8)
    for fam in family_list
]
ax.legend(handles=legend_elements, title="Model Family", fontsize=8, title_fontsize=9, loc="upper right")

ax.set_xlabel("Normalized Training Time")
ax.set_ylabel("1-Wasserstein Distance")
ax.set_title("Model Performance: 1-Wasserstein vs. Normalized Training Time")

plt.tight_layout()
plt.show()

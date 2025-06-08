import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.optim as optim
import gpytorch
import matplotlib
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


"_______________Loading Data___________________"
#If your data is in GSLIB format
def read_gslib(filename: str):
    with open(filename, "r") as f:
        lines = f.readlines()
        ncols = int(lines[1].split()[0])
        col_names = [lines[i + 2].strip() for i in range(ncols)]
        df = pd.read_csv(filename, skiprows=ncols + 2, delim_whitespace=True, names=col_names)
        return df

# Data in well locations
file = read_gslib(filename="SeismicTraining_forPhieUK")
data = file.to_numpy()
inputs = data[:, 7:]
outputs = data[:, 6]
x_coord = data[:, 3]
y_coord = data[:, 4]
z_coord = data[:, 5]

"_________________Shuffle________________________"

# Generate a random permutation for shuffling
per_list = np.random.permutation(len(data))

# Shuffle inputs, outputs, and coordinates
inputs_sh = inputs[per_list]
outputs_sh = outputs[per_list]
x_coord_sh = x_coord[per_list]
y_coord_sh = y_coord[per_list]
z_coord_sh = z_coord[per_list]

"__________________Normalize Data___________________"

min_vec = inputs_sh.min(axis=0)
max_vec = inputs_sh.max(axis=0)
inputs_sh = (inputs_sh - min_vec) / (max_vec - min_vec)

min_vecx = x_coord_sh.min(axis=0)
max_vecx = x_coord_sh.max(axis=0)
x_coord_sh = (x_coord_sh - min_vecx) / (max_vecx - min_vecx)

min_vecy = y_coord_sh.min(axis=0)
max_vecy = y_coord_sh.max(axis=0)
y_coord_sh = (y_coord_sh - min_vecy) / (max_vecy - min_vecy)

min_vecz = z_coord_sh.min(axis=0)
max_vecz = z_coord_sh.max(axis=0)
z_coord_sh = (z_coord_sh - min_vecz) / (max_vecz - min_vecz)

# Split the data into inputs and outputs
x_train, x_test, y_train, y_test = train_test_split(inputs_sh, outputs_sh, test_size=0.25, random_state=42)

# Split the coordinates based on the same indices
x_train_coord, x_test_coord = train_test_split(x_coord_sh, test_size=0.25, random_state=42)
y_train_coord, y_test_coord = train_test_split(y_coord_sh, test_size=0.25, random_state=42)
z_train_coord, z_test_coord = train_test_split(z_coord_sh, test_size=0.25, random_state=42)


# Fit a Gaussian Process on the predictions from Random Forest
class GaussianProcessModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GaussianProcessModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()  # Using RBF kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_rf_gp_model(x_train, y_train, x_test, y_test, x_train_coord, y_train_coord, z_train_coord,
                      x_test_coord, y_test_coord, z_test_coord, n_estimators=80, gp_epochs=1500):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf_model.fit(x_train, y_train)

    # Predict with RF
    tree_predictions_train = np.column_stack([tree.predict(x_train) for tree in rf_model.estimators_])
    tree_predictions_test = np.column_stack([tree.predict(x_test) for tree in rf_model.estimators_])

    # Add coordinates
    train_combined = np.concatenate((tree_predictions_train,
                                     x_train_coord[:, None], y_train_coord[:, None], z_train_coord[:, None]), axis=1)
    test_combined = np.concatenate((tree_predictions_test,
                                    x_test_coord[:, None], y_test_coord[:, None], z_test_coord[:, None]), axis=1)

    train_combined_tensor = torch.FloatTensor(train_combined)
    test_combined_tensor = torch.FloatTensor(test_combined)
    y_train_tensor = torch.FloatTensor(y_train).view(-1)

    # GP model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = GaussianProcessModel(train_combined_tensor, y_train_tensor, likelihood)

    gp_model.train()
    likelihood.train()
    optimizer = optim.Adam(gp_model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    for epoch in range(gp_epochs):
        optimizer.zero_grad()
        output = gp_model(train_combined_tensor)
        loss = -mll(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluation
    gp_model.eval()
    likelihood.eval()
    with torch.no_grad():
        y_pred_test = gp_model(test_combined_tensor).mean.detach().cpu().numpy()
        y_pred_train = gp_model(train_combined_tensor).mean.detach().cpu().numpy()

    metrics = {
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
        'mse_train': mse(y_train, y_pred_train),
        'mse_test': mse(y_test, y_pred_test),
        'y_test_true': y_test,
        'y_test_pred': y_pred_test
    }

    return metrics

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)
r2_train_list, r2_test_list, mse_train_list, mse_test_list = [], [], [], []
all_true, all_pred = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs_sh)):
    x_train, x_test = inputs_sh[train_idx], inputs_sh[test_idx]
    y_train, y_test = outputs_sh[train_idx], outputs_sh[test_idx]
    x_train_coord, x_test_coord = x_coord_sh[train_idx], x_coord_sh[test_idx]
    y_train_coord, y_test_coord = y_coord_sh[train_idx], y_coord_sh[test_idx]
    z_train_coord, z_test_coord = z_coord_sh[train_idx], z_coord_sh[test_idx]

    metrics = train_rf_gp_model(x_train, y_train, x_test, y_test,
                                x_train_coord, y_train_coord, z_train_coord,
                                x_test_coord, y_test_coord, z_test_coord)

    print(f"Fold {fold+1} - R² test: {metrics['r2_test']:.3f}, MSE test: {metrics['mse_test']:.3f}")
    r2_train_list.append(metrics['r2_train'])
    r2_test_list.append(metrics['r2_test'])
    mse_train_list.append(metrics['mse_train'])
    mse_test_list.append(metrics['mse_test'])

    all_true.extend(metrics['y_test_true'])
    all_pred.extend(metrics['y_test_pred'])

# Average metrics
print(f"\nAverage R² test: {np.mean(r2_test_list):.3f} ± {np.std(r2_test_list):.3f}")
print(f"Average MSE test: {np.mean(mse_test_list):.3f} ± {np.std(mse_test_list):.3f}")

plt.figure(figsize=(6,6))
plt.scatter(all_true, all_pred, c='blue', alpha=0.5, edgecolor='k')
plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'r--')
plt.xlabel('True Phie')
plt.ylabel('Predicted Phie')
plt.title(f'{k}-fold Cross Plot \nR² = {np.mean(r2_test_list):.2f}, MSE = {np.mean(mse_test_list):.4f}')
plt.grid(False)
plt.tight_layout()
plt.show()

plt.savefig("K-fold.png")


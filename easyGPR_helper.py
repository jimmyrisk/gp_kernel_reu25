import torch
import gpytorch
import numpy as np
import pandas as pd
from gpytorch.constraints import Positive
from copy import deepcopy

# -----------------------------------------------------------------------------
# Device & dtype utilities
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device():
    """Return the global default torch device (CPU or CUDA)."""
    return DEVICE


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def to_numpy(tensor: torch.Tensor):
    """Detach (if needed), move to CPU, convert to NumPy array."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input to to_numpy must be a torch.Tensor.")
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy()


def to_torch(x, *, device=None, dtype=None):
    """Convert array‑like `x` to a torch.Tensor on the requested device/dtype."""
    device = get_device() if device is None else device
    dtype = torch.get_default_dtype() if dtype is None else dtype

    # Early exit for existing tensors
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)

    # NumPy / pandas
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=device, dtype=dtype)

    # Python scalars / lists / tuples
    return torch.tensor(x, device=device, dtype=dtype)


# -----------------------------------------------------------------------------
# Optional global GPyTorch settings (unchanged API)
# -----------------------------------------------------------------------------

def set_gpytorch_settings(dtype=torch.float64, use_cuda: bool = True):
    """Optional knob‑twiddling.  Code is device‑agnostic without this call."""
    gpytorch.settings.fast_computations.covar_root_decomposition._set_state(False)
    gpytorch.settings.fast_computations.log_prob._set_state(False)
    gpytorch.settings.fast_computations.solves._set_state(False)
    gpytorch.settings.cholesky_max_tries._set_value(100)
    gpytorch.settings.debug._set_state(False)
    gpytorch.settings.min_fixed_noise._set_value(1e-7, 1e-7, 1e-7)

    torch.set_default_dtype(dtype)

    if use_cuda and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    elif use_cuda and not torch.cuda.is_available():
        import warnings
        warnings.warn("CUDA requested but not available — running on CPU.")

    gpytorch.settings.ciq_samples._set_state(False)
    gpytorch.settings.skip_logdet_forward._set_state(False)
    gpytorch.settings.num_trace_samples._set_value(0)
    gpytorch.settings.num_gauss_hermite_locs._set_value(300)
    gpytorch.settings.num_likelihood_samples._set_value(300)
    gpytorch.settings.deterministic_probes._set_state(True)


# -----------------------------------------------------------------------------
# Scaling utilities
# -----------------------------------------------------------------------------
class MinMaxScaler:
    """Min‑max scaling to [0,1] along each feature column."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mins: torch.Tensor | None = None
        self.maxs: torch.Tensor | None = None

    # --------------------------- public API ---------------------------
    def fit(self, X):
        self.reset()
        X = to_torch(X)
        self.mins = torch.min(X, dim=0).values
        self.maxs = torch.max(X, dim=0).values
        return self.scale(X)

    def scale(self, X):
        X = to_torch(X)
        return (X - self.mins) / (self.maxs - self.mins)

    def unscale(self, X_scaled):
        X_scaled = to_torch(X_scaled)
        X = X_scaled.unsqueeze(-1) * (self.maxs - self.mins) + self.mins
        return X.squeeze(-1)


class NoScale:
    """No‑op scaler with the same interface as MinMaxScaler."""

    @staticmethod
    def fit(_):
        pass

    @staticmethod
    def scale(X):
        return to_torch(X)

    @staticmethod
    def unscale(X_scaled):
        return to_torch(X_scaled)


# -----------------------------------------------------------------------------
# GP Regression model
# -----------------------------------------------------------------------------
class GPRModel(gpytorch.models.ExactGP):
    """Exact GP regression with optional feature scaling and device safety."""

    def __init__(
        self,
        train_x=None,
        train_y=None,
        *,
        kernel: gpytorch.kernels.Kernel,
        mean: str | gpytorch.means.Mean = "constant",
        scale_x: bool = True,
    ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Positive())

        # Scaling utility
        self.scaler = MinMaxScaler() if scale_x else NoScale()

        # Prepare training data (may be None => pure prior GP)
        if train_x is None and train_y is None:
            x_scaled = None
            y_torch = None
        else:
            x_torch = to_torch(train_x)
            y_torch = to_torch(train_y)
            self.scaler.fit(x_torch)
            x_scaled = self.scaler.scale(x_torch)

        self.train_x = train_x if train_x is None else x_torch
        self.train_y = train_y if train_y is None else y_torch
        self.train_x_scaled = x_scaled

        super().__init__(x_scaled, y_torch, likelihood)

        # Mean module
        if mean == "constant":
            self.mean_module = gpytorch.means.ConstantMean()
        elif mean is None:
            self.mean_module = gpytorch.means.ZeroMean()
        elif isinstance(mean, gpytorch.means.Mean):
            self.mean_module = deepcopy(mean)
        else:
            raise TypeError("'mean' must be 'constant', None, or a gpytorch.means.Mean.")

        # Kernel
        self.kernel = deepcopy(kernel)

        # Finalise device placement
        self.to(get_device())

        self.predictions: Predictions | None = None

    # -------------------------------- forward -------------------------------
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.kernel(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # -------------------------------- fitting -------------------------------
    def fit_model(self, training_iterations: int = 50, lr: float = 0.1, verbose: bool = True):
        self.train()
        self.likelihood.train()

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for _ in range(training_iterations):
            opt.zero_grad()
            output = self(self.train_x_scaled)
            loss = -mll(output, self.train_y)
            loss.backward()
            opt.step()

        self.compute_bic()

        if verbose:
            print("Fitting complete.")
            print(f"--- final mll: {-loss:.4f}")
            print(f"--- num_params: {self.num_param}")
            print(f"--- BIC: {self.bic:.4f}")
        return self

    # -------------------------------- prediction ---------------------------
    def make_predictions(
        self,
        test_x,
        *,
        type: str = "f",  # 'f' = latent, 'y' = noisy observations
        return_type: str = "numpy",
        posterior: bool = True,
    ):
        if posterior:
            self.eval()
            self.likelihood.eval()
        else:
            self.train()
            self.likelihood.train()

        test_x = to_torch(test_x)
        test_x_scaled = self.scaler.scale(test_x)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if type == "f":
                preds = self(test_x_scaled)
            elif type == "y":
                preds = self.likelihood(self(test_x_scaled))
            else:
                raise ValueError("type must be 'f' or 'y'.")

        self.predictions = Predictions(preds.mean, preds.variance)
        if return_type == "numpy":
            self.predictions.to_numpy()
        return self.predictions

    # -------------------------------- simulation ---------------------------
    def simulate(
        self,
        x_sim,
        *,
        method: str = "prior",  # 'prior' | 'posterior'
        type: str = "f",  # 'f' | 'y'
        return_type: str = "numpy",
        n_paths: int = 1,
    ):
        if n_paths > 1:
            raise NotImplementedError("call simulate repeatedly for multiple paths.")
        if method not in {"prior", "posterior"}:
            raise ValueError("method must be 'prior' or 'posterior'.")
        if type not in {"f", "y"}:
            raise ValueError("type must be 'f' or 'y'.")

        x_sim = to_torch(x_sim)
        x_scaled = self.scaler.scale(x_sim)

        # Switch GP mode
        if method == "prior":
            self.train()
        else:
            self.eval()

        with torch.no_grad():
            base_dist = self(x_scaled)
            if type == "f":
                samples = base_dist.rsample()
            else:  # 'y'
                samples = self.likelihood(base_dist).rsample()

        if return_type == "numpy":
            samples = to_numpy(samples)
        return samples

    # -------------------------------- diagnostics --------------------------
    def get_LOOCV(self):
        self.train()
        y_dist = self.likelihood(self(self.train_x_scaled))
        K = y_dist.covariance_matrix
        y = self.train_y

        K_inv = torch.inverse(K)
        y_hat = y - (K_inv @ y.unsqueeze(-1)).squeeze() / torch.diag(K_inv)
        rmse = (y - y_hat).pow(2).mean().sqrt()
        return rmse

    def compute_bic(self, data=None):
        self.train()
        X = self.train_x_scaled if data is None else to_torch(data)
        n = X.shape[0]

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self).to(X.device)
        log_marg_like = mll(self(X), self.train_y)

        self.num_param = sum(p.numel() for p in self.parameters())
        with torch.no_grad():
            self.bic = -log_marg_like * n + self.num_param * np.log(n) / 2
        return self.bic

    def get_hyperparameters_df(self, *, scaled: bool = True):
        rows: list[dict] = []
        for name, param in self.named_hyperparameters():
            clean = strip_raw_prefix(name)
            constraint = self.constraint_for_parameter_name(name)
            value = constraint.transform(param) if constraint is not None else param

            if not scaled and "lengthscale" in clean:
                value = value * (self.scaler.maxs - self.scaler.mins)

            arr = to_numpy(value)
            if arr.size > 1:
                for v in arr.flatten():
                    rows.append({"Hyperparameter": clean, "Estimate": v})
            else:
                rows.append({"Hyperparameter": clean, "Estimate": arr.item()})
        return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Auxiliary containers / utils
# -----------------------------------------------------------------------------
class Predictions:
    """Lightweight container for predictive mean and variance."""

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    # Mutating conversions for convenience ------------------------------------------------
    def to_numpy(self):
        self.mean = to_numpy(self.mean)
        self.variance = to_numpy(self.variance)

    def to_torch(self):
        self.mean = to_torch(self.mean)
        self.variance = to_torch(self.variance)


def strip_raw_prefix(name: str) -> str:
    """Remove gpytorch's "raw_" prefix from parameter names for readability."""
    parts = name.split(".")
    if parts[-1].startswith("raw_"):
        parts[-1] = parts[-1][4:]
    return ".".join(parts)

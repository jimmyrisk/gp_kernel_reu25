# Existing content
import torch
import gpytorch
from gpytorch.kernels.kernel import Kernel
from gpytorch.priors import Prior
from gpytorch.constraints import Interval, Positive
from gpytorch.lazy import MatmulLazyTensor, LowRankRootLazyTensor
from typing import Optional
import pandas as pd

def to_numpy(tensor):
    if tensor.requires_grad:
        if torch.cuda.is_available():
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()
    else:
        if torch.cuda.is_available():
            return tensor.cpu().numpy()
        else:
            return tensor.numpy()

def to_torch(array, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.Tensor(array.astype(np.float32)).to(device)


#from settings import set_gpytorch_settings
#set_gpytorch_settings()


class SigmaKernel(Kernel):
    r"""
    Nonstationary kernel:
        k(x, x') = σ(x) σ(x'),
    with σ(x) = exp( -α * f(x) ),
    where f(x)=|x| for `power='abs'` or f(x)=x² for `power='square'`.
    
    This implementation delays materializing full tensors by using lazy tensors.
    """
    def __init__(self, power='abs', alpha_prior: Optional[object] = None, alpha_constraint=None, **kwargs):
        super(SigmaKernel, self).__init__(**kwargs)
        self.power = power
        if alpha_constraint is None:
            alpha_constraint = Positive()
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(1)))
        if alpha_prior is not None:
            self.register_prior("alpha_prior", alpha_prior,
                                lambda m: m.alpha,
                                lambda m, v: m._set_alpha(v))
        self.register_constraint("raw_alpha", alpha_constraint)

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)
    
    @alpha.setter
    def alpha(self, value):
        self._set_alpha(value)
    
    def _set_alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=self.raw_alpha.dtype, device=self.raw_alpha.device)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))
    
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # Lazify inputs similar to MinKernel
        x1_ = x1
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
        
        # Compute σ(x₁) elementwise
        if self.power == 'abs':
            sigma1 = torch.exp(-self.alpha * torch.abs(x1_))
        elif self.power == 'square':
            sigma1 = torch.exp(-self.alpha * (x1_ ** 2))
        else:
            raise ValueError("`power` must be either 'abs' or 'square'.")
        
        if diag:
            # Return the diagonal (σ(x)²) without extra lazy ops
            return sigma1.squeeze(-1) ** 2
        
        # If inputs are identical, use a low-rank representation for efficiency
        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Squeeze and then unsqueeze to ensure shape (..., n, 1)
            return LowRankRootLazyTensor(sigma1.squeeze(-1).unsqueeze(-1))
        else:
            # Process x2 similarly
            x2_ = x2
            if last_dim_is_batch:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)
            if self.power == 'abs':
                sigma2 = torch.exp(-self.alpha * torch.abs(x2_))
            elif self.power == 'square':
                sigma2 = torch.exp(-self.alpha * (x2_ ** 2))
            # Use MatmulLazyTensor to represent the outer product lazily:
            # sigma1 is shape (..., n, 1) and sigma2 becomes (..., m, 1) after squeeze/unsqueeze.
            return MatmulLazyTensor(
                sigma1.squeeze(-1).unsqueeze(-1),
                sigma2.squeeze(-1).unsqueeze(-1).transpose(-2, -1)
            )



class SpaceTime1d(Kernel):
    def __init__(self, **kwargs):
        super(SpaceTime1d, self).__init__(**kwargs)

        # Unconstrained parameter
        self.register_parameter(
            name="raw_epsilon",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )

        # Parameters constrained to be non-negative
        self.register_parameter(
            name="raw_a",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_a", Positive())

        self.register_parameter(
            name="raw_a_prime",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_a_prime", Positive())

        self.register_parameter(
            name="raw_c",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_c", Positive())

        self.register_parameter(
            name="raw_phi",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_phi", Positive())

        self.register_parameter(
            name="raw_alpha",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_alpha", Positive())

        self.register_parameter(
            name="raw_delta",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_delta", Positive())

        # Parameters constrained to be in [0,1]
        self.register_parameter(
            name="raw_alpha_prime",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_alpha_prime", Interval(0.0, 1.0))

        self.register_parameter(
            name="raw_gamma",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_gamma", Interval(0.0, 1.0))

    def forward(self, x1, x2, diag=False, **params):
        
        x1_ = x1.index_select(-1, torch.tensor([0]))
        t1_ = x1.index_select(-1, torch.tensor([1]))
        if x2 is not None:
            x2_ = x2.index_select(-1, torch.tensor([0]))
            t2_ = x2.index_select(-1, torch.tensor([1]))

        # Give x1_ and x2_ a last dimension, if necessary
        if x1_.ndimension() == 1:
            x1_ = x1_.unsqueeze(1)
            t1_ = t1_.unsqueeze(1)
        if x2_ is not None:
            if x2_.ndimension() == 1:
                x2_ = x2_.unsqueeze(1)
                t2_ = t2_.unsqueeze(1)
            if not x1_.size(-1) == x2_.size(-1):
                raise RuntimeError("x1_ and x2_ must have the same number of dimensions!")

        if x2_ is None:
            x2_ = x1_
            t2_ = t1_

        epsilon = self.epsilon
        phi = self.phi
        a = self.a
        a_prime = self.a_prime
        c = self.c
        alpha = self.alpha
        delta = self.delta
        alpha_prime = self.alpha_prime
        gamma = self.gamma

        # Compute distance matrices using Euclidean distance
        #x_dist = self.covar_dist(x1_, x2_ + epsilon * (t2_ - t1_).abs(), diag=diag, **params)


        # Compute the indicator matrix where spatial components are equal
        # Ensure that x1_ and x2_ have compatible shapes
        a = torch.ones(x1_.shape)

        b = torch.ones(x2_.shape)
        aa = MatmulLazyTensor(x1_, b.transpose(-2, -1)).evaluate()
        bb = MatmulLazyTensor(a, x2_.transpose(-2, -1)).evaluate()

        indicator = torch.eq(aa,bb)*1.0

        x_dist = self.covar_dist(x1_ - epsilon*t1_, 
                                 x2_ - epsilon*t2_, 
                                 diag=diag, **params)
        
        t_dist = self.covar_dist(t1_, t2_, 
                                 diag=diag, **params)

        term1 = phi.div((1+a*t_dist.pow(2)).pow(alpha))
        
        term2 = (-c.mul(x_dist.pow(2*gamma)).div((1+a*t_dist.pow(2)).pow(alpha*gamma))).exp()
        
        term3 = delta * indicator.div((1+a_prime*t_dist.pow(2)).pow(alpha_prime))
        
        K = term1 * term2 + term3

        if diag:
            return K.diag()
        else:
            return K

    @property
    def epsilon(self):
        return self.raw_epsilon

    @epsilon.setter
    def epsilon(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_epsilon)
        self.initialize(raw_epsilon=value)

    @property
    def a(self):
        return self.raw_a_constraint.transform(self.raw_a)

    @a.setter
    def a(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_a)
        self.initialize(raw_a=self.raw_a_constraint.inverse_transform(value))

    @property
    def a_prime(self):
        return self.raw_a_prime_constraint.transform(self.raw_a_prime)

    @a_prime.setter
    def a_prime(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_a_prime)
        self.initialize(raw_a_prime=self.raw_a_prime_constraint.inverse_transform(value))

    @property
    def c(self):
        return self.raw_c_constraint.transform(self.raw_c)

    @c.setter
    def c(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_c)
        self.initialize(raw_c=self.raw_c_constraint.inverse_transform(value))

    @property
    def phi(self):
        return self.raw_phi_constraint.transform(self.raw_phi)

    @phi.setter
    def phi(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_phi)
        self.initialize(raw_phi=self.raw_phi_constraint.inverse_transform(value))

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    @property
    def delta(self):
        return self.raw_delta_constraint.transform(self.raw_delta)

    @delta.setter
    def delta(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_delta)
        self.initialize(raw_delta=self.raw_delta_constraint.inverse_transform(value))

    @property
    def alpha_prime(self):
        return self.raw_alpha_prime_constraint.transform(self.raw_alpha_prime)

    @alpha_prime.setter
    def alpha_prime(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha_prime)
        self.initialize(raw_alpha_prime=self.raw_alpha_prime_constraint.inverse_transform(value))

    @property
    def gamma(self):
        return self.raw_gamma_constraint.transform(self.raw_gamma)

    @gamma.setter
    def gamma(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gamma)
        self.initialize(raw_gamma=self.raw_gamma_constraint.inverse_transform(value))



class MinKernel(Kernel):

    def __init__(
        self,
        offset_prior: Optional[Prior] = None,
        offset_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        super(MinKernel, self).__init__(**kwargs)
        if offset_constraint is None:
            offset_constraint = Interval(0.001, 100, initial_value=0.01)
        self.register_parameter(name="raw_offset", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))

        if offset_prior is not None:
            if not isinstance(offset_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(offset_prior).__name__)
            self.register_prior("offset_prior", offset_prior, lambda m: m.offset, lambda m, v: m._set_offset(v))

        self.register_constraint("raw_offset", offset_constraint)

    @property
    def offset(self) -> torch.Tensor:
        return self.raw_offset_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value: torch.Tensor) -> None:
        self._set_offset(value)

    def _set_offset(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_offset)
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1_ = x1
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
        a = torch.ones(x1_.shape)
        # a.to(x1_.device)  # for cuda vs cpu  (I don't think this works - may need fixing later)
        offset = self.offset.view(*self.batch_shape, 1, 1)

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLazyTensor when x1 == x2 for efficiency when composing
            # with other kernels
            aa = MatmulLazyTensor(x1_, a.transpose(-2, -1))
            bb = aa.transpose(-2,-1)
            K = 0.5 * (aa + bb - torch.abs((aa - bb).evaluate())) + offset

        else:
            x2_ = x2
            if last_dim_is_batch:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)
            b = torch.ones(x2_.shape)
            # b.to(x2_.device)  # for cuda vs cpu  (I don't think this works - may need fixing later)
            aa = MatmulLazyTensor(x1_, b.transpose(-2, -1))
            bb = MatmulLazyTensor(a, x2_.transpose(-2, -1))
            K = 0.5 * (aa + bb - torch.abs((aa - bb).evaluate())) + offset
        if diag:
            return K.diag()
        else:
            return K


class FractionalGaussianNoise(Kernel):
    has_lengthscale = True
    is_stationary = False

    def __init__(
            self,
            H_prior: Optional[Prior] = None,
            H_constraint: Optional[Interval] = None,
            **kwargs
    ):
        
        super(FractionalGaussianNoise, self).__init__(**kwargs)
        if H_constraint is None:
            #H_constraint = Interval(1e-5, 1 - 1e-5, initial_value=0.9)
            H_constraint = Interval(0.51, 0.99, initial_value=0.75)

        ard_num_dims = kwargs.get("ard_num_dims", 1)
        self.register_parameter(
            name="raw_H", parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1, ard_num_dims))
        )

        if H_prior is not None:
            if not isinstance(H_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(H_prior).__name__)
            self.register_prior(
                "H_prior",
                H_prior,
                lambda m: m.H,
                lambda m, v: m._set_H(v),
            )

        self.register_constraint("raw_H", H_constraint)

    @property
    def H(self):
        return self.raw_H_constraint.transform(self.raw_H)

    @H.setter
    def H(self, value):
        self._set_H(value)

    def _set_H(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_H)
        self.initialize(raw_H=self.raw_H_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):

        alpha = 2 * self.H

        x1_ = (x1).div(self.lengthscale)
        x2_ = (x2).div(self.lengthscale)        

        a = torch.ones(x1_.shape)

        b = torch.ones(x2_.shape)
        aa = MatmulLazyTensor(x1_, b.transpose(-2, -1))
        bb = MatmulLazyTensor(a, x2_.transpose(-2, -1))
        h = aa - bb
        h = h.evaluate()

        term1 = torch.abs(h + 1).pow(alpha)
        term2 = -2 * torch.abs(h).pow(alpha)
        term3 = torch.abs(h - 1).pow(alpha)
        res = (term1 + term2 + term3).div(2)

        return res

import torch
import torch.nn as nn
import math
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive, Interval

class RBFNonSeparableKernel(Kernel):
    """
    Custom RBF Kernel that uses a lengthscale matrix A^{-1} for computing
    (x - y)^T A^{-1} (x - y).
    Specifically tailored for 2D inputs with parameterization in terms of a, c, and rho.
    """
    has_lengthscale = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize raw parameters for a, c, and rho
        self.register_parameter(name="raw_a", parameter=nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_parameter(name="raw_c", parameter=nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_parameter(name="raw_rho", parameter=nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        
        # Apply constraints
        self.register_constraint("raw_a", Positive())
        self.register_constraint("raw_c", Positive())
        self.register_constraint("raw_rho", Interval(-1 + 1e-5, 1 - 1e-5))  # rho in (-1, 1)

    @property
    def a(self):
        return self.raw_a_constraint.transform(self.raw_a)
    
    @property
    def c(self):
        return self.raw_c_constraint.transform(self.raw_c)
    
    @property
    def rho(self):
        return self.raw_rho_constraint.transform(self.raw_rho)
    
    def forward(self, x1, x2, diag=False, **params):
        # Retrieve parameters and remove extra dimensions
        a = self.a.squeeze().to(dtype=x1.dtype, device=x1.device)
        c = self.c.squeeze().to(dtype=x1.dtype, device=x1.device)
        rho = self.rho.squeeze().to(dtype=x1.dtype, device=x1.device)
    
        # Compute elements of A^{-1}
        sqrt_ac = torch.sqrt(a * c)
        A_inv = torch.stack([
            torch.stack([a, -rho * sqrt_ac], dim=0),
            torch.stack([-rho * sqrt_ac, c], dim=0)
        ], dim=0)  # Shape: [2, 2]
        
        if diag:
            # Diagonal elements are always 1 in RBF since distance is zero
            return torch.ones(x1.size(0), device=x1.device, dtype=x1.dtype)
        
        # Compute the quadratic form (x - y)^T A^{-1} (x - y) for all pairs
        # x1: (n, 2)
        # x2: (m, 2)
        # diff: (n, m, 2)
        diff = x1.unsqueeze(1) - x2.unsqueeze(0)  # Shape: (n, m, 2)
        
        # Reshape diff to (n*m, 2)
        n, m, _ = diff.shape
        diff_flat = diff.reshape(-1, 2)  # Shape: (n*m, 2)
        
        # Compute diff_Ainv = diff @ A_inv
        diff_Ainv_flat = diff_flat @ A_inv  # Shape: (n*m, 2)
        
        # Compute the quadratic form (x - y)^T A^{-1} (x - y)
        sq_dist_flat = torch.sum(diff_flat * diff_Ainv_flat, dim=1)  # Shape: (n*m,)
        
        # Reshape back to (n, m)
        sq_dist = sq_dist_flat.view(n, m)
        
        # Apply the RBF formula
        return torch.exp(-0.5 * sq_dist)

    
    def get_original_hyperparameters(self):
        """
        Retrieve the original hyperparameters ell1, ell2, and rho from a, c, and rho.
        
        Returns:
            dict: {
                'ell1': ell1,
                'ell2': ell2,
                'rho': rho
            }
        """
        a = self.a
        c = self.c
        rho = self.rho
        one_minus_rho_sq = 1.0 - rho ** 2
        
        # Compute ell1 and ell2
        ell1 = torch.sqrt(1.0 / (a * one_minus_rho_sq))
        ell2 = torch.sqrt(1.0 / (c * one_minus_rho_sq))
        
        return {
            'ell1': ell1.item(),
            'ell2': ell2.item(),
            'rho': rho.item()
        }
    
    def __repr__(self):
        original_hyperparams = self.get_original_hyperparameters()
        return (f"{self.__class__.__name__}(ell1={original_hyperparams['ell1']:.4f}, "
                f"ell2={original_hyperparams['ell2']:.4f}, rho={original_hyperparams['rho']:.4f})")

class MaxDistanceKernel(Kernel):
    r"""
    Custom kernel computing
        k(x, y) = c - ||x - y||,
    where c is the maximal possible distance (the domain's diameter).
    Note: c must be set appropriately as a fixed hyperparameter.
    """
    has_lengthscale = False

    def __init__(self, c: float, **kwargs):
        """
        Args:
            c (float): The maximal distance in the domain. Should be chosen such that
                       for all x, y in the domain, ||x-y|| <= c.
        """
        super().__init__(**kwargs)
        self.c = c  # fixed hyperparameter
        self.gamma = 0.5  # anisotropy parameter

        self.A = torch.stack([
            torch.stack([torch.tensor(1.0), torch.tensor(-self.gamma)], dim=0),
            torch.stack([torch.tensor(-self.gamma), torch.tensor(1.0)], dim=0)
        ], dim=0)  # Shape: [2, 2]

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        if diag:
            # When x1 == x2, ||x - y|| = 0, so k(x, x) = c.
            return torch.full((x1.size(0),), self.c, dtype=x1.dtype, device=x1.device)

        diff = x1.unsqueeze(1) - x2.unsqueeze(0)  # Shape: (n, m, 2)
        
        # Reshape diff to (n*m, 2)
        n, m, _ = diff.shape
        diff_flat = diff.reshape(-1, 2)  # Shape: (n*m, 2)
        
        # Compute diff_Ainv = diff @ A_inv
        diff_Ainv_flat = diff_flat @ self.A  # Shape: (n*m, 2)
        
        # Compute the quadratic form (x - y)^T A^{-1} (x - y)
        sq_dist_flat = torch.sum(diff_flat * diff_Ainv_flat, dim=1)  # Shape: (n*m,)
        
        # Reshape back to (n, m)
        sq_dist = sq_dist_flat.view(n, m)

        return self.c - sq_dist.sqrt()


    def get_original_hyperparameters(self):

        return {
            'c': self.c,
            'gamma': self.gamma
        }
    
    def __repr__(self):
        original_hyperparams = self.get_original_hyperparameters()
        return (f"{self.__class__.__name__}(c={original_hyperparams['c']:.4f},gamma={original_hyperparams['gamma']:.4f})")


# class RBFNonSeparableKernel(Kernel):
#     """
#     Custom RBF Kernel that uses a lengthscale matrix A^{-1} for computing
#     (x - y)^T A^{-1} (x - y).
#     Specifically tailored for 2D inputs.
#     """
#     has_lengthscale = True

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
        
#         # Initialize raw parameters for lengthscales and correlation
#         self.register_parameter(name="raw_lengthscale1", parameter=torch.nn.Parameter(torch.randn(1)))
#         self.register_parameter(name="raw_lengthscale2", parameter=torch.nn.Parameter(torch.randn(1)))
#         self.register_parameter(name="raw_rho", parameter=torch.nn.Parameter(torch.randn(1)))
        
#         # Apply constraints
#         self.register_constraint("raw_lengthscale1", Positive())
#         self.register_constraint("raw_lengthscale2", Positive())
#         self.register_constraint("raw_rho", Interval(-1 + 1e-5, 1 - 1e-5))  # Allow rho to be in (-1, 1)

#     @property
#     def lengthscale1(self):
#         return self.raw_lengthscale1_constraint.transform(self.raw_lengthscale1)
    
#     @property
#     def lengthscale2(self):
#         return self.raw_lengthscale2_constraint.transform(self.raw_lengthscale2)
    
#     @property
#     def rho(self):
#         return self.raw_rho_constraint.transform(self.raw_rho)
    
#     def forward(self, x1, x2, diag=False, **params):
#         """
#         Compute the RBF kernel matrix using A^{-1} where A is defined as:
        
#             A = [[l1^2, l1*l2*rho],
#                  [l1*l2*rho, l2^2]]
        
#         Therefore, A^{-1} is computed as:
        
#             A^{-1} = (1 / (l1^2*l2^2*(1 - rho^2))) * [[l2^2, -l1*l2*rho],
#                                                        [-l1*l2*rho, l1^2]]
#         """
#         # Retrieve parameters
#         l1 = self.lengthscale1  # (1,)
#         l2 = self.lengthscale2  # (1,)
#         rho = self.rho           # (1,)

#         # Compute determinant of A
#         det_A = (l1 ** 2) * (l2 ** 2) * (1 - rho ** 2)
        
#         # Ensure determinant is positive to maintain positive definiteness
#         # (Handled implicitly by constraints on rho and positive lengthscales)
        
#         # Compute A^{-1}
#         A_inv = torch.empty(2, 2, device=x1.device, dtype=x1.dtype)
#         A_inv[0, 0] = (l2 ** 2) / det_A
#         A_inv[0, 1] = (-l1 * l2 * rho) / det_A
#         A_inv[1, 0] = (-l1 * l2 * rho) / det_A
#         A_inv[1, 1] = (l1 ** 2) / det_A

#         if diag:
#             # If diag=True, compute the diagonal elements directly
#             # (x - y)^T A^{-1} (x - y) simplifies to sum over i,j of (x_i - y_i) A^{-1}_{i,j} (x_j - y_j)
#             # When computing diagonals, x1 and x2 are the same, so (x - y) = 0, but diag=True implies x1 and x2 are identical.
#             # However, to handle diag=True properly, it's better to compute 0 or the diagonal elements.
#             # In standard RBF with A^{-1}, diag would be all ones since distance is zero.
#             return torch.ones(x1.size(0), device=x1.device, dtype=x1.dtype)
        
#         # Compute the quadratic form (x - y)^T A^{-1} (x - y) for all pairs
#         # x1: n x 2
#         # x2: m x 2
#         # We want to compute for each i, j: (x1[i] - x2[j])^T A^{-1} (x1[i] - x2[j})
        
#         # Expand dimensions to compute pairwise differences
#         # (n x 1 x 2) - (1 x m x 2) = (n x m x 2)
#         diff = x1.unsqueeze(1) - x2.unsqueeze(0)  # (n x m x 2)
        
#         # Compute the quadratic form using batch matrix multiplication
#         # First multiply diff by A_inv: (n x m x 2) * (2 x 2) -> (n x m x 2)
#         diff_Ainv = torch.matmul(diff, A_inv)  # (n x m x 2)
        
#         # Element-wise multiply and sum over the last dimension to get scalar quadratic form
#         sq_dist = torch.sum(diff_Ainv * diff, dim=2)  # (n x m)
        
#         # Apply the RBF formula
#         return torch.exp(-0.5 * sq_dist)



import torch
import torch.nn as nn
import math
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive, Interval



import math
import torch
from torch import nn
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive, Interval
from typing import Optional


class MaternNonSeparableKernel(Kernel):
    """
    Custom 2D anisotropic Matern kernel with Mahalanobis distance.
    Supports nu in {0.5, 1.5, 2.5}.
    Parameters a, c, rho define the inverse lengthscale matrix A_inv.
    """
    # Remove if not using default GPyTorch "lengthscale" logic.
    # has_lengthscale = True

    def __init__(self, nu: float = 2.5, **kwargs):
        super().__init__(**kwargs)
        if nu not in {0.5, 1.5, 2.5}:
            raise ValueError("nu must be one of {0.5, 1.5, 2.5}")
        self.nu = nu

        # Raw parameters
        self.register_parameter("raw_a", nn.Parameter(torch.randn(1)))
        self.register_parameter("raw_c", nn.Parameter(torch.randn(1)))
        self.register_parameter("raw_rho", nn.Parameter(torch.randn(1)))

        # Constraints
        self.register_constraint("raw_a", Positive())
        self.register_constraint("raw_c", Positive())
        self.register_constraint("raw_rho", Interval(-0.99999, 0.99999))

    @property
    def a(self):
        return self.raw_a_constraint.transform(self.raw_a)

    @property
    def c(self):
        return self.raw_c_constraint.transform(self.raw_c)

    @property
    def rho(self):
        return self.raw_rho_constraint.transform(self.raw_rho)

    def _eval_kernel(self, sq_dist: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Matern kernel on the squared distances.

        -  nu=0.5: Exp(-r)
        -  nu=1.5: (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
        -  nu=2.5: (1 + sqrt(5)*r + 5/3 * r^2) * exp(-sqrt(5)*r)
        """
        r = torch.sqrt(sq_dist + 1e-20)

        if self.nu == 0.5:
            return torch.exp(-r)
        elif self.nu == 1.5:
            sqrt3 = math.sqrt(3)
            return (1.0 + sqrt3 * r) * torch.exp(-sqrt3 * r)
        elif self.nu == 2.5:
            sqrt5 = math.sqrt(5)
            return (1.0 + sqrt5 * r + (5.0 / 3.0) * sq_dist) * torch.exp(-sqrt5 * r)
        else:
            raise ValueError("Unsupported nu (should be in {0.5,1.5,2.5}).")

    def _sq_mahalanobis_dist(self, x1: torch.Tensor, x2: torch.Tensor, diag=False) -> torch.Tensor:
        """
        (x - y)^T A_inv (x - y), with A_inv determined by a, c, rho.
        """
        a_val = self.a.squeeze()
        c_val = self.c.squeeze()
        rho_val = self.rho.squeeze()

        # Build the A_inv matrix
        import math
        sqrt_ac = torch.sqrt(a_val * c_val)
        A_inv = torch.stack([
            torch.stack([a_val, -rho_val * sqrt_ac], dim=0),
            torch.stack([-rho_val * sqrt_ac, c_val], dim=0)
        ], dim=0)  # shape (2, 2)

        if diag:
            # Return zeros for the diagonal distance
            return torch.zeros(x1.size(0), device=x1.device, dtype=x1.dtype)

        diff = x1.unsqueeze(1) - x2.unsqueeze(0)  # shape (n, m, 2)
        diff_flat = diff.reshape(-1, 2)
        diff_Ainv_flat = diff_flat @ A_inv
        sq_dist_flat = (diff_flat * diff_Ainv_flat).sum(dim=1)
        return sq_dist_flat.view(diff.size(0), diff.size(1))

    def forward(self, x1, x2, diag=False, **params):
        sq_dist = self._sq_mahalanobis_dist(x1, x2, diag=diag)
        return self._eval_kernel(sq_dist)


    
    def get_original_hyperparameters(self):
        """
        Retrieve the original hyperparameters ell1, ell2, and rho from a, c, and rho.
        
        Returns:
            dict: {
                'ell1': ell1,
                'ell2': ell2,
                'rho': rho
            }
        """
        a = self.a
        c = self.c
        rho = self.rho
        one_minus_rho_sq = 1.0 - rho ** 2
        
        # Compute ell1 and ell2
        ell1 = torch.sqrt(1.0 / (a * one_minus_rho_sq))
        ell2 = torch.sqrt(1.0 / (c * one_minus_rho_sq))
        
        return {
            'ell1': ell1.item(),
            'ell2': ell2.item(),
            'rho': rho.item()
        }
    
    def __repr__(self):
        original_hyperparams = self.get_original_hyperparameters()
        return (f"{self.__class__.__name__}(nu={self.nu}, "
                f"ell1={original_hyperparams['ell1']:.4f}, "
                f"ell2={original_hyperparams['ell2']:.4f}, rho={original_hyperparams['rho']:.4f})")
    
    def print_original_hyperparameters(self):
        """
        Prints the original hyperparameters ell1, ell2, and rho.
        """
        hyperparams = self.get_original_hyperparameters()
        print(f"ell1: {hyperparams['ell1']:.4f}, ell2: {hyperparams['ell2']:.4f}, rho: {hyperparams['rho']:.4f}")




# class MaternNonSeparableKernel(Kernel):
#     """
#     Custom non-separable Matern kernel tailored for 2D inputs.
#     Supports nu=0.5, 1.5, and 2.5.
#     """
#     has_lengthscale = True

#     def __init__(self, nu: Optional[float] = 2.5, **kwargs):
#         if nu not in {0.5, 1.5, 2.5}:
#             raise ValueError("nu must be one of 0.5, 1.5, or 2.5")
#         super().__init__(**kwargs)
#         self.nu = nu

#         # Initialize raw parameters for lengthscales and correlation
#         self.register_parameter(name="raw_lengthscale1", parameter=nn.Parameter(torch.randn(1)))
#         self.register_parameter(name="raw_lengthscale2", parameter=nn.Parameter(torch.randn(1)))
#         self.register_parameter(name="raw_rho", parameter=nn.Parameter(torch.randn(1)))

#         # Apply constraints
#         self.register_constraint("raw_lengthscale1", Positive())
#         self.register_constraint("raw_lengthscale2", Positive())
#         self.register_constraint("raw_rho", Interval(-1 + 1e-5, 1 - 1e-5))  # rho in (-1, 1)

#     @property
#     def lengthscale1(self):
#         return self.raw_lengthscale1_constraint.transform(self.raw_lengthscale1)

#     @property
#     def lengthscale2(self):
#         return self.raw_lengthscale2_constraint.transform(self.raw_lengthscale2)

#     @property
#     def rho(self):
#         return self.raw_rho_constraint.transform(self.raw_rho)

#     def forward(self, x1, x2, diag=False, **params):
#         """
#         Compute the Matern kernel matrix using A^{-1} where A is defined as:

#             A = [[l1^2, l1*l2*rho],
#                  [l1*l2*rho, l2^2]]

#         Therefore, A^{-1} is computed as:

#             A^{-1} = (1 / (l1^2*l2^2*(1 - rho^2))) * [[l2^2, -l1*l2*rho],
#                                                      [-l1*l2*rho, l1^2]]
#         """
#         # Retrieve parameters
#         l1 = self.lengthscale1  # Scalar tensor
#         l2 = self.lengthscale2  # Scalar tensor
#         rho = self.rho           # Scalar tensor

#         # Compute determinant of A
#         det_A = (l1 ** 2) * (l2 ** 2) * (1 - rho ** 2)

#         # Compute A^{-1}
#         A_inv = torch.empty(2, 2, device=x1.device, dtype=x1.dtype)
#         A_inv[0, 0] = (l2 ** 2) / det_A
#         A_inv[0, 1] = (-l1 * l2 * rho) / det_A
#         A_inv[1, 0] = (-l1 * l2 * rho) / det_A
#         A_inv[1, 1] = (l1 ** 2) / det_A

#         if diag:
#             # When diag=True, return ones as the distance is zero
#             return torch.ones(x1.size(0), device=x1.device, dtype=x1.dtype)

#         # Compute the quadratic form (x - y)^T A^{-1} (x - y) for all pairs
#         # x1: n x 2
#         # x2: m x 2
#         # We want to compute for each i, j: (x1[i] - x2[j])^T A^{-1} (x1[i] - x2[j})

#         # Expand dimensions to compute pairwise differences
#         # (n x 1 x 2) - (1 x m x 2) = (n x m x 2)
#         diff = x1.unsqueeze(1) - x2.unsqueeze(0)  # Shape: (n, m, 2)

#         # Compute the quadratic form using batch matrix multiplication
#         # First multiply diff by A_inv: (n x m x 2) * (2 x 2) -> (n x m x 2)
#         diff_Ainv = torch.matmul(diff, A_inv)  # Shape: (n, m, 2)

#         # Element-wise multiply and sum over the last dimension to get scalar quadratic form
#         sq_dist = torch.sum(diff_Ainv * diff, dim=2)  # Shape: (n, m)

#         # Compute the scaled distance with numerical stability
#         scaled_distance = torch.sqrt(sq_dist + 1e-12)  # Shape: (n, m)

#         # Apply the Matern formula based on the value of nu
#         if self.nu == 0.5:
#             # Matern 1/2: k(r) = exp(-r)
#             return torch.exp(-scaled_distance)
#         elif self.nu == 1.5:
#             # Matern 3/2: k(r) = (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
#             sqrt3 = math.sqrt(3)
#             sqrt3_r = sqrt3 * scaled_distance
#             return (1.0 + sqrt3_r) * torch.exp(-sqrt3_r)
#         elif self.nu == 2.5:
#             # Matern 5/2: k(r) = (1 + sqrt(5)*r + (5/3)*r^2) * exp(-sqrt(5)*r)
#             sqrt5 = math.sqrt(5)
#             sqrt5_r = sqrt5 * scaled_distance
#             return (1.0 + sqrt5_r + (5.0 / 3.0) * sq_dist) * torch.exp(-sqrt5_r)
#         else:
#             raise ValueError("Unsupported nu value.")

#     def __repr__(self):
#         return f"{self.__class__.__name__}(nu={self.nu})"



# Existing content
import torch
import gpytorch
from gpytorch.kernels.kernel import Kernel
from gpytorch.priors import Prior
from gpytorch.constraints import Interval, Positive
from gpytorch.lazy import MatmulLazyTensor, LowRankRootLazyTensor
from typing import Optional
import pandas as pd
class fBMKernel(Kernel):
    has_lengthscale = False

    def __init__(
            self,
            H_prior: Optional[Prior] = None,
            H_constraint: Optional[Interval] = None,
            **kwargs
    ):
        super(fBMKernel, self).__init__(**kwargs)
        if H_constraint is None:
            H_constraint = Interval(0.51, 0.99, initial_value=0.75)

        ard_num_dims = kwargs.get("ard_num_dims", 1)
        self.register_parameter(
            name="raw_H", parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1, ard_num_dims))
        )

        if H_prior is not None:
            if not isinstance(H_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(H_prior).__name__)
            self.register_prior(
                "H_prior",
                H_prior,
                lambda m: m.H,
                lambda m, v: m._set_H(v),
            )

        self.register_constraint("raw_H", H_constraint)

    @property
    def H(self):
        return self.raw_H_constraint.transform(self.raw_H)

    @H.setter
    def H(self, value):
        self._set_H(value)

    def _set_H(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_H)
        self.initialize(raw_H=self.raw_H_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):

        alpha = 2*self.H

        a = torch.ones(x1.shape)

        b = torch.ones(x2.shape)
        aa = MatmulLazyTensor(x1, b.transpose(-2, -1))
        bb = MatmulLazyTensor(a, x2.transpose(-2, -1))
        h = aa - bb
        h = h.evaluate()
        
        term1 = torch.abs(aa.evaluate()).pow(alpha)
        term2 = -torch.abs(h).pow(alpha)
        term3 = torch.abs(bb.evaluate()).pow(alpha)
        res =  (term1 + term2 + term3).div(2)

        return res




class WendlandKernel(Kernel):
    r"""
    Computes a compactly supported Wendland kernel.
    
    For inputs x1 and x2, let t = distance(x1,x2)/lengthscale.
    Then the kernel is defined as:
    
        k(t) = (1-t)_+^p * poly(t)
    
    where
        for k = 0:   p = μ, 
                     poly(t) = 1,
        for k = 1:   p = μ+1,
                     poly(t) = 1 + (μ+1)*t,
        for k = 2:   p = μ+2,
                     poly(t) = 1 + (μ+2)*t + t^2/3*(μ^2+4μ+3).
                     
    The kernel is zero for t >= 1.
    
    Args:
        mu (float, default=1.0): Positive hyperparameter μ.
        k (int, default=0): Order of the kernel. Must be 0, 1, or 2.
    """
    has_lengthscale = True
    is_stationary = True

    def __init__(self, mu: float = 1.0, k: int = 0, **kwargs):
        super(WendlandKernel, self).__init__(**kwargs)
        if k not in {0, 1, 2}:
            raise ValueError("k must be one of {0, 1, 2}.")
        self.k = k

        # Register raw_mu so that we can constrain it to be positive.
        self.register_parameter("raw_mu", nn.Parameter(torch.tensor(mu)))
        self.register_constraint("raw_mu", Positive())

    @property
    def mu(self):
        return self.raw_mu_constraint.transform(self.raw_mu)
    
    @mu.setter
    def mu(self, value):
        self._set_mu(value)

    def _set_mu(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mu)
        self.initialize(raw_mu=self.raw_mu_constraint.inverse_transform(value))


    def forward(self, x1, x2, diag=False, **params):
        # Subtract the mean for numerical stability (does not affect stationarity)
        mean = x1.mean(dim=-2, keepdim=True)
        x1_ = (x1 - mean).div(self.lengthscale)
        x2_ = (x2 - mean).div(self.lengthscale)
        # Compute the unitless distance t = (||x1 - x2|| / lengthscale)
        t = self.covar_dist(x1_, x2_, diag=diag, **params)
        mu = self.mu
        ell = 0 + self.k + 1 + mu
        # Define the compact support (1-t)_+ = max(1-t,0)
        support = (1 - t).clamp(min=0)
        if self.k == 0:
            result = support.pow(ell)
        elif self.k == 1:
            result = support.pow(ell+1) * (1+(ell+1)*t)
        elif self.k == 2:
            result = support.pow(ell+2) * (
                (ell**2 + 4*ell + 3) * t.pow(2) + (3*ell + 6)*t + 3
            )        
        return result

    def get_original_hyperparameters(self):
        """
        Retrieve the original hyperparameters: lengthscale and μ.
        
        Returns:
            dict: {'lengthscale': ..., 'mu': ...}
        """
        # Assuming lengthscale is a scalar parameter.
        lengthscale = self.lengthscale.item() if self.lengthscale.numel() == 1 else self.lengthscale
        mu = self.mu.item() if self.mu.numel() == 1 else self.mu
        return {"lengthscale": lengthscale, "mu": mu}

    def __repr__(self):
        hyperparams = self.get_original_hyperparameters()
        return (f"{self.__class__.__name__}(k={self.k}, mu={hyperparams['mu']:.4f}, "
                f"lengthscale={hyperparams['lengthscale']:.4f})")




class KernelWrapper:
    def __init__(self):
        self.kernel_instance = None
        self.original_train_x = None

    def create_kernel(self, kernel_type, ard_num_dims=None, scaler=None, **kwargs):
        self.scaler = scaler
        if isinstance(kernel_type, str):
            kernel_type = kernel_type.lower()
        if not isinstance(kernel_type, str):
            # todo: better handling of custom kernels
            self.kernel_instance = kernel_type
            return self.kernel_instance
        elif kernel_type in ["rbf", "squared exponential", "gaussian"]:
            # The RBF kernel has many names
            self.kernel_instance = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims, **kwargs)
        elif kernel_type in ["exp", "exponential", "laplace", "mat12", "m12"]:
            # The exponential kernel has many names
            self.kernel_instance = gpytorch.kernels.MaternKernel(ard_num_dims=ard_num_dims, nu=0.5, **kwargs)
        elif kernel_type.startswith("mat"):
            if kernel_type == "mat32":
                nu_value = 1.5
            elif kernel_type == "mat52":
                nu_value = 2.5
            else:
                nu_value = float(kernel_type.split("_")[1])
            self.kernel_instance = gpytorch.kernels.MaternKernel(ard_num_dims=ard_num_dims, nu=nu_value, **kwargs)
        elif kernel_type.startswith("lin"):
            self.kernel_instance = gpytorch.kernels.PolynomialKernel(power = 1, ard_num_dims=ard_num_dims, **kwargs)
        elif kernel_type.startswith("min"):
            self.kernel_instance = MinKernel(ard_num_dims=ard_num_dims, **kwargs)

        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        # todo: handle kernel addition and multiplication
        self.kernel_instance = gpytorch.kernels.ScaleKernel(self.kernel_instance)
        return self.kernel_instance


    def get_kernel_instance(self):
        return self.kernel_instance

    def print_hyperparameters(self, feature_names=None, verbose=False):
        # todo: beta1_orig = beta1 / (x_max - x_min); beta0_orig = beta0 - beta1_orig * x_min
        data = []
        for name, param in self.kernel_instance.named_hyperparameters():
            # Getting the constraint corresponding to the current hyperparameter
            constraint = self.kernel_instance.constraint_for_parameter_name(name)

            # Getting the raw value of the parameter
            raw_value = param.data

            # Getting the transformed value (according to GPyTorch)
            transformed_value = constraint.transform(param)

            # If it is a lengthscale, apply min/max unscaling
            if 'lengthscale' in name:
                unscaled_value = transformed_value * (self.scaler.maxs - self.scaler.mins)
            else:
                unscaled_value = transformed_value  # For non-lengthscale parameters, no unscaling is applied

            # Converting to numpy for easier handling
            unscaled_numpy = to_numpy(unscaled_value)
            transformed_numpy = to_numpy(transformed_value)
            raw_numpy = to_numpy(raw_value)

            # Preparing data for dataframe
            if unscaled_numpy.size > 1:  # Case where the parameter is a vector
                for idx, value in enumerate(unscaled_numpy.flatten()):
                    feature_name = feature_names[idx] if feature_names is not None else None

                    entry = {
                        "Hyperparameter Name": name,
                        "Feature Name": feature_name,
                        "Unscaled Value": value,
                        "Scaled Value": transformed_numpy.flatten()[idx]
                    }

                    if verbose:
                        entry.update({
                            "Raw Value (GPyTorch)": raw_numpy[idx],
                            "Constraint": str(constraint)
                        })

                    data.append(entry)
            else:  # Case where the parameter is a scalar
                entry = {
                    "Hyperparameter Name": name,
                    "Feature Name": None,
                    "Unscaled Value": unscaled_numpy.item(),
                    "Scaled Value": transformed_numpy.item()
                }

                if verbose:
                    entry.update({
                        "Raw Value (GPyTorch)": raw_numpy.item(),
                        "Constraint": str(constraint)
                    })

                data.append(entry)

        # Creating dataframe
        df = pd.DataFrame(data)
        return df


from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive, Interval
from gpytorch.lazy import MatmulLazyTensor
class SpaceTime1d(Kernel):
    def __init__(self, **kwargs):
        super(SpaceTime1d, self).__init__(**kwargs)

        # Unconstrained parameter
        self.register_parameter(
            name="raw_epsilon",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )

        # Parameters constrained to be non-negative
        self.register_parameter(
            name="raw_a",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_a", Positive())

        self.register_parameter(
            name="raw_a_prime",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_a_prime", Positive())

        self.register_parameter(
            name="raw_c",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_c", Positive())

        self.register_parameter(
            name="raw_phi",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_phi", Positive())

        self.register_parameter(
            name="raw_alpha",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_alpha", Positive())

        self.register_parameter(
            name="raw_delta",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_delta", Positive())

        # Parameters constrained to be in [0,1]
        self.register_parameter(
            name="raw_alpha_prime",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_alpha_prime", Interval(0.0, 1.0))

        self.register_parameter(
            name="raw_gamma",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        self.register_constraint("raw_gamma", Interval(0.0, 1.0))

    def forward(self, x1, x2, diag=False, **params):
        
        x1_ = x1.index_select(-1, torch.tensor([0], device=x1.device))
        t1_ = x1.index_select(-1, torch.tensor([1], device=x1.device))
        if x2 is not None:
            x2_ = x2.index_select(-1, torch.tensor([0], device=x1.device))
            t2_ = x2.index_select(-1, torch.tensor([1], device=x1.device))

        # Give x1_ and x2_ a last dimension, if necessary
        if x1_.ndimension() == 1:
            x1_ = x1_.unsqueeze(1)
            t1_ = t1_.unsqueeze(1)
        if x2_ is not None:
            if x2_.ndimension() == 1:
                x2_ = x2_.unsqueeze(1)
                t2_ = t2_.unsqueeze(1)
            if not x1_.size(-1) == x2_.size(-1):
                raise RuntimeError("x1_ and x2_ must have the same number of dimensions!")

        if x2_ is None:
            x2_ = x1_
            t2_ = t1_

        epsilon = self.epsilon
        phi = self.phi
        a = self.a
        a_prime = self.a_prime
        c = self.c
        alpha = self.alpha
        delta = self.delta
        alpha_prime = self.alpha_prime
        gamma = self.gamma

        # Compute distance matrices using Euclidean distance
        #x_dist = self.covar_dist(x1_, x2_ + epsilon * (t2_ - t1_).abs(), diag=diag, **params)


        # Compute the indicator matrix where spatial components are equal
        # Ensure that x1_ and x2_ have compatible shapes
        a = torch.ones(x1_.shape, device=x1.device)

        b = torch.ones(x2_.shape, device=x1.device)
        aa = MatmulLazyTensor(x1_, b.transpose(-2, -1)).evaluate()
        bb = MatmulLazyTensor(a, x2_.transpose(-2, -1)).evaluate()

        indicator = torch.eq(aa,bb)*1.0

        x_dist = self.covar_dist(x1_ - epsilon*t1_, 
                                 x2_ - epsilon*t2_, 
                                 diag=diag, **params)
        
        t_dist = self.covar_dist(t1_, t2_, 
                                 diag=diag, **params)

        term1 = phi.div((1+a*t_dist.pow(2)).pow(alpha))
        
        term2 = (-c.mul(x_dist.pow(2*gamma)).div((1+a*t_dist.pow(2)).pow(alpha*gamma))).exp()
        
        term3 = delta * indicator.div((1+a_prime*t_dist.pow(2)).pow(alpha_prime))
        
        K = term1 * term2 + term3

        if diag:
            return K.diag()
        else:
            return K

    @property
    def epsilon(self):
        return self.raw_epsilon

    @epsilon.setter
    def epsilon(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_epsilon)
        self.initialize(raw_epsilon=value)

    @property
    def a(self):
        return self.raw_a_constraint.transform(self.raw_a)

    @a.setter
    def a(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_a)
        self.initialize(raw_a=self.raw_a_constraint.inverse_transform(value))

    @property
    def a_prime(self):
        return self.raw_a_prime_constraint.transform(self.raw_a_prime)

    @a_prime.setter
    def a_prime(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_a_prime)
        self.initialize(raw_a_prime=self.raw_a_prime_constraint.inverse_transform(value))

    @property
    def c(self):
        return self.raw_c_constraint.transform(self.raw_c)

    @c.setter
    def c(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_c)
        self.initialize(raw_c=self.raw_c_constraint.inverse_transform(value))

    @property
    def phi(self):
        return self.raw_phi_constraint.transform(self.raw_phi)

    @phi.setter
    def phi(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_phi)
        self.initialize(raw_phi=self.raw_phi_constraint.inverse_transform(value))

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    @property
    def delta(self):
        return self.raw_delta_constraint.transform(self.raw_delta)

    @delta.setter
    def delta(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_delta)
        self.initialize(raw_delta=self.raw_delta_constraint.inverse_transform(value))

    @property
    def alpha_prime(self):
        return self.raw_alpha_prime_constraint.transform(self.raw_alpha_prime)

    @alpha_prime.setter
    def alpha_prime(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha_prime)
        self.initialize(raw_alpha_prime=self.raw_alpha_prime_constraint.inverse_transform(value))

    @property
    def gamma(self):
        return self.raw_gamma_constraint.transform(self.raw_gamma)

    @gamma.setter
    def gamma(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gamma)
        self.initialize(raw_gamma=self.raw_gamma_constraint.inverse_transform(value))


# New kernel definitions
class RBFKernel(gpytorch.kernels.RBFKernel):
    def __init__(self):
        super().__init__()
        self.lengthscale = torch.tensor([1.0])

class LinearKernel(gpytorch.kernels.LinearKernel):
    def __init__(self):
        super().__init__()
        self.variance = torch.tensor([1.0])

class PeriodicKernel(gpytorch.kernels.PeriodicKernel):
    def __init__(self):
        super().__init__()

class ExponentialKernel(gpytorch.kernels.MaternKernel):
    def __init__(self):
        super().__init__(nu=0.5)
        self.lengthscale = torch.tensor([1.0])

class ProductKernel(gpytorch.kernels.ProductKernel):
    def __init__(self, kern1, kern2):
        super().__init__(kern1, kern2)


class PowerExponentialKernel(Kernel):
    """
    Power Exponential Kernel defined as exp(-(r/l)^alpha).

    * r is the distance between inputs x1 and x2.
    * l is a lengthscale parameter.
    * alpha is a parameter satisfying 0 < alpha <= 2.
    """

    has_lengthscale = True

    def __init__(
            self,
            alpha_prior: Optional[Prior] = None,
            alpha_constraint: Optional[Interval] = None,
            **kwargs
    ):
        super(PowerExponentialKernel, self).__init__(**kwargs)
        if alpha_constraint is None:
            alpha_constraint = Interval(1e-5, 2+1e-5, initial_value=1.5)

        ard_num_dims = kwargs.get("ard_num_dims", 1)
        self.register_parameter(
            name="raw_alpha", parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1, ard_num_dims))
        )

        if alpha_prior is not None:
            if not isinstance(alpha_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(alpha_prior).__name__)
            self.register_prior(
                "alpha_prior",
                alpha_prior,
                lambda m: m.alpha,
                lambda m, v: m._set_alpha(v),
            )

        self.register_constraint("raw_alpha", alpha_constraint)

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        self._set_alpha(value)

    def _set_alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        # Calculate the distance
        distance = self.covar_dist(x1, x2, diag=diag, **params)
        # Apply the Power Exponential kernel function
        return torch.exp(-torch.pow(distance.div(self.lengthscale), self.alpha))



class GeneralCauchy(Kernel):

    has_lengthscale = True

    def __init__(self, alpha_constraint: Optional[Interval] = None, beta_constraint: Optional[Interval] = None, **kwargs):
        super(GeneralCauchy, self).__init__(**kwargs)
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        if alpha_constraint is None:
            alpha_constraint = Interval(0,2)

        self.register_constraint("raw_alpha", alpha_constraint)

        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        if beta_constraint is None:
            beta_constraint = Positive()

        self.register_constraint("raw_beta", beta_constraint)

    def forward(self, x1, x2, diag=False, **params):
        def postprocess_rq(dist_mat):
            alpha = self.alpha
            beta = self.beta
            for _ in range(1, len(dist_mat.shape) - len(self.batch_shape)):
                alpha = alpha.unsqueeze(-1)
                beta = beta.unsqueeze(-1)
            return (1 + dist_mat.pow(alpha)).pow(-beta.div(alpha))

        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        return postprocess_rq(
            self.covar_dist(x1_, x2_, square_dist=False, diag=diag, **params),
        )

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    @property
    def beta(self):
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))


import torch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Interval, Positive
from torch import Tensor
from typing import Optional

class GeneralizedOU(Kernel):
    """
    Generalized Ornstein-Uhlenbeck (GeneralizedOU) Kernel.

    k(h) = cosh(H * |h/l|) - [sinh(|h/l| / 2)]^{2*H} / 2

    where:
        - 0 < H < 1
        - l > 0 (lengthscale)
    """

    has_lengthscale = True
    is_stationary = True

    def __init__(
        self,
        H_constraint: Optional[Interval] = None,
        **kwargs
    ):
        """
        Initializes the GeneralizedOU kernel.

        Args:
            H_constraint (Optional[Interval]): Constraint on the H parameter.
                Defaults to Interval(1e-5, 1 - 1e-5) to ensure 0 < H < 1.
            **kwargs: Additional keyword arguments for the Kernel base class.
        """
        super(GeneralizedOU, self).__init__(**kwargs)

        # Define constraint for H: 0 < H < 1
        if H_constraint is None:
            H_constraint = Interval(1e-5, 1 - 1e-5)

        # Register raw_H parameter
        self.register_parameter(name="raw_H", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))

        # Register H constraint
        self.register_constraint("raw_H", H_constraint)

        self.lengthscale = 1.0

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        """
        Computes the covariance matrix using the GeneralizedOU kernel.

        Args:
            x1 (Tensor): First input tensor of shape (n, d).
            x2 (Tensor): Second input tensor of shape (m, d).
            diag (bool, optional): If True, only the diagonal elements are computed.
                Defaults to False.
            **params: Additional parameters.

        Returns:
            Tensor: Covariance matrix of shape (n, m) or (n,) if diag=True.
        """
        x1_ = (x1).div(self.lengthscale)
        x2_ = (x2).div(self.lengthscale)
        distance = self.covar_dist(x1_, x2_, diag=diag, **params)

        # Retrieve H
        H = self.H

        # Compute each term of the kernel
        term1 = torch.cosh(H * distance)
        term2 = (torch.sinh(distance / 2)*2).pow(2 * H) / 2

        # Kernel function
        k = term1 - term2

        return k

    @property
    def H(self) -> Tensor:
        """
        Returns the transformed H parameter ensuring it satisfies the constraint.

        Returns:
            Tensor: H parameter with shape compatible with the kernel's batch shape.
        """
        return self.raw_H_constraint.transform(self.raw_H)

    @H.setter
    def H(self, value: Tensor) -> None:
        """
        Sets the H parameter by transforming it back to the raw parameter space.

        Args:
            value (Tensor): New value for H parameter.
        """
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_H)

        self.initialize(raw_H=self.raw_H_constraint.inverse_transform(value))


class TwoParamKernel(Kernel):
    """
    Two-Parameter Kernel.

    k(s, t) = (s * t)^H * exp(-alpha * |log(t / s)|)

    where:
        - 0 < H < 1
        - alpha > 0
    """

    has_lengthscale = False
    is_stationary = False  # Non-stationary kernel

    def __init__(
        self,
        H_constraint: Optional[Interval] = None,
        alpha_constraint: Optional[Positive] = None,
        **kwargs
    ):
        """
        Initializes the TwoParamKernel.

        Args:
            H_constraint (Optional[Interval]): Constraint on the H parameter.
                Defaults to Interval(1e-5, 1 - 1e-5) to ensure 0 < H < 1.
            alpha_constraint (Optional[Positive]): Constraint on the alpha parameter.
                Defaults to Positive() to ensure alpha > 0.
            **kwargs: Additional keyword arguments for the Kernel base class.
        """
        super(TwoParamKernel, self).__init__(**kwargs)

        # Define constraint for H: 0 < H < 1
        if H_constraint is None:
            H_constraint = Interval(1e-5, 1 - 1e-5)

        # Define constraint for alpha: alpha > 0
        if alpha_constraint is None:
            alpha_constraint = Positive()

        # Register raw_H parameter
        self.register_parameter(name="raw_H", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))

        # Register H constraint
        self.register_constraint("raw_H", H_constraint)

        # Register raw_alpha parameter
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))

        # Register alpha constraint
        self.register_constraint("raw_alpha", alpha_constraint)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        """
        Computes the covariance matrix using the Two-Parameter Kernel.

        Args:
            x1 (Tensor): First input tensor of shape (n, d).
            x2 (Tensor): Second input tensor of shape (m, d).
            diag (bool, optional): If True, only the diagonal elements are computed.
                Defaults to False.
            **params: Additional parameters.

        Returns:
            Tensor: Covariance matrix of shape (n, m) or (n,) if diag=True.
        """
        # Ensure inputs are positive to avoid log(0) or negative values
        if torch.any(x1 <= 0) or torch.any(x2 <= 0):
            raise ValueError("All input values must be positive.")

        # Compute (s * t)^H
        # s has shape (n, d), t has shape (m, d)
        # Compute element-wise multiplication and then power
        product = x1.unsqueeze(1) * x2.unsqueeze(0)  # Shape: (n, m, d)
        product_pow_H = product.pow(self.H)  # Shape: (n, m, d)

        # Compute |log(t / s)| = |log(t) - log(s)|
        log_ratio = torch.log(x2.unsqueeze(0)) - torch.log(x1.unsqueeze(1))  # Shape: (n, m, d)
        abs_log_ratio = torch.abs(log_ratio)  # Shape: (n, m, d)

        # Compute exp(-alpha * |log(t / s)|)
        exp_term = torch.exp(-self.alpha * abs_log_ratio)  # Shape: (n, m, d)

        # Combine both terms
        k = product_pow_H * exp_term  # Shape: (n, m, d)

        # Sum over the last dimension to handle multi-dimensional inputs
        k = k.sum(dim=-1)  # Shape: (n, m)

        # If diag=True, return the diagonal
        if diag:
            return k.diagonal(dim1=-2, dim2=-1)

        return k

    @property
    def H(self) -> Tensor:
        """
        Returns the transformed H parameter ensuring it satisfies the constraint.

        Returns:
            Tensor: H parameter with shape compatible with the kernel's batch shape.
        """
        return self.raw_H_constraint.transform(self.raw_H)

    @H.setter
    def H(self, value: Tensor) -> None:
        """
        Sets the H parameter by transforming it back to the raw parameter space.

        Args:
            value (Tensor): New value for H parameter.
        """
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_H)
        self.initialize(raw_H=self.raw_H_constraint.inverse_transform(value))

    @property
    def alpha(self) -> Tensor:
        """
        Returns the transformed alpha parameter ensuring it satisfies the constraint.

        Returns:
            Tensor: alpha parameter with shape compatible with the kernel's batch shape.
        """
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value: Tensor) -> None:
        """
        Sets the alpha parameter by transforming it back to the raw parameter space.

        Args:
            value (Tensor): New value for alpha parameter.
        """
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))



class NonStationaryGeneralizedOU(Kernel):
    """
    Non-Stationary Generalized Ornstein-Uhlenbeck (OU) Kernel.

    k(x, y) = cosh((H + alpha * t)|tau|) - [2 * sinh(|tau| / 2)]^(H + alpha * t)

    where:
        - x = t + tau / 2
        - y = t - tau / 2
        - 0 < H < 1
        - alpha > 0
    """

    has_lengthscale = False
    is_stationary = False  # Non-stationary kernel

    def __init__(
        self,
        H_constraint: Optional[Interval] = None,
        alpha_constraint: Optional[Positive] = None,
        **kwargs
    ):
        """
        Initializes the Non-Stationary Generalized OU kernel.

        Args:
            H_constraint (Optional[Interval]): Constraint on the H parameter.
                Defaults to Interval(1e-5, 1 - 1e-5) to ensure 0 < H < 1.
            alpha_constraint (Optional[Positive]): Constraint on the alpha parameter.
                Defaults to Positive() to ensure alpha > 0.
            **kwargs: Additional keyword arguments for the Kernel base class.
        """
        super(NonStationaryGeneralizedOU, self).__init__(**kwargs)

        # Define constraint for H: 0 < H < 1
        if H_constraint is None:
            H_constraint = Interval(1e-5, 1 - 1e-5)

        # Define constraint for alpha: alpha > 0
        if alpha_constraint is None:
            alpha_constraint = Positive()

        # Register raw_H parameter
        self.register_parameter(name="raw_H", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))

        # Register H constraint
        self.register_constraint("raw_H", H_constraint)

        # Register raw_alpha parameter
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))

        # Register alpha constraint
        self.register_constraint("raw_alpha", alpha_constraint)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params):
        """
        Computes the covariance matrix using the Non-Stationary Generalized OU kernel.

        Args:
            x1 (Tensor): First input tensor of shape (n, d).
            x2 (Tensor): Second input tensor of shape (m, d).
            diag (bool, optional): If True, only the diagonal elements are computed.
                Defaults to False.
            **params: Additional parameters.

        Returns:
            TensorLazyTensor: Covariance matrix as a LazyTensor of shape (n, m).
        """
        # Ensure inputs are positive to avoid log(0) or negative values
        if torch.any(x1 <= 0) or torch.any(x2 <= 0):
            raise ValueError("All input values must be positive.")

        # Compute t and tau
        # Given x = t + tau / 2 and y = t - tau / 2,
        # solve for t and tau:
        # t = (x + y) / 2
        # tau = x - y

        # Reshape x1 and x2 to (n, m, d)
        t = (x1.unsqueeze(1) + x2.unsqueeze(0)) / 2  # Shape: (n, m, d)
        tau = x1.unsqueeze(1) - x2.unsqueeze(0)       # Shape: (n, m, d)

        # Compute (H + alpha * t)
        H_plus_alpha_t = self.H + self.alpha * t      # Shape: (n, m, d)

        # Compute |tau|
        abs_tau = torch.abs(tau)                      # Shape: (n, m, d)

        # Compute cosh((H + alpha * t) * |tau|)
        term1 = torch.cosh(H_plus_alpha_t * abs_tau) # Shape: (n, m, d)

        # Compute [2 * sinh(|tau| / 2)]^(H + alpha * t)
        # First compute 2 * sinh(|tau| / 2)
        sinh_term = 2 * torch.sinh(abs_tau / 2)       # Shape: (n, m, d)

        # To prevent numerical issues, add a small epsilon inside the power
        epsilon = 1e-8
        sinh_term = sinh_term + epsilon

        # Compute sinh_term raised to the power (H + alpha * t)
        term2 = torch.pow(sinh_term, H_plus_alpha_t)  # Shape: (n, m, d)

        # Compute the kernel matrix
        k = term1 - term2                               # Shape: (n, m, d)

        # Since we assume one-dimensional input, sum over the last dimension
        k = k.sum(dim=-1)                               # Shape: (n, m)


        # If diag=True, return the diagonal
        if diag:
            return k.diagonal(dim1=-2, dim2=-1)

        return k

    @property
    def H(self) -> Tensor:
        """
        Returns the transformed H parameter ensuring it satisfies the constraint.

        Returns:
            Tensor: H parameter with shape compatible with the kernel's batch shape.
        """
        return self.raw_H_constraint.transform(self.raw_H)

    @H.setter
    def H(self, value: Tensor) -> None:
        """
        Sets the H parameter by transforming it back to the raw parameter space.

        Args:
            value (Tensor): New value for H parameter.
        """
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_H)
        self.initialize(raw_H=self.raw_H_constraint.inverse_transform(value))

    @property
    def alpha(self) -> Tensor:
        """
        Returns the transformed alpha parameter ensuring it satisfies the constraint.

        Returns:
            Tensor: alpha parameter with shape compatible with the kernel's batch shape.
        """
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value: Tensor) -> None:
        """
        Sets the alpha parameter by transforming it back to the raw parameter space.

        Args:
            value (Tensor): New value for alpha parameter.
        """
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))
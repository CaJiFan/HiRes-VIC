import math
import torch

def spd_cholesky_map(params: torch.Tensor) -> torch.Tensor:
    """
    Maps a 6D Euclidean vector to an SPD Stiffness Matrix K.
    Method: Cholesky Factorization K = L @ L.T
    
    Input: params (batch_size, 6) -> [d1, d2, d3, o1, o2, o3]
            - d: diagonal elements (must be positive)
            - o: off-diagonal elements
    Output: K (batch_size, 3, 3)
    """
    batch_size = params.shape[0]
    L = torch.zeros((batch_size, 3, 3), device=params.device)
    
    # Map first 3 params to diagonal (using Softplus to ensure positivity > 0)
    # This enforces the SPD constraint structurally.
    diag_indices = torch.arange(3)
    L[:, diag_indices, diag_indices] = torch.nn.functional.softplus(params[:, :3]) + 0.1 # Min stiffness bias
    
    # Map last 3 params to lower triangle
    L[:, 1, 0] = params[:, 3]
    L[:, 2, 0] = params[:, 4]
    L[:, 2, 1] = params[:, 5]
    
    # Compute K = L @ L.T
    K = torch.bmm(L, L.transpose(1, 2))
    return K

def spd_grl_map(params: torch.Tensor) -> torch.Tensor:
    """
    Exact implementation of the G-RL SPD mapping.
    Mandel Basis -> Symmetric Matrix -> Matrix Exponential -> SPD Matrix
    """
    batch_size = params.shape[0]
    S = torch.zeros((batch_size, 3, 3), device=params.device)
    
    # Diagonals
    S[:, 0, 0] = params[:, 0]
    S[:, 1, 1] = params[:, 1]
    S[:, 2, 2] = params[:, 2]
    
    # Off-diagonals (Scaled by 1/sqrt(2) to preserve the Mandel isometric mapping)
    # Note: The NN outputs the Mandel coordinates, we scale them back to matrix entries
    inv_sqrt2 = 1.0 / math.sqrt(2.0) 
    
    S[:, 0, 1] = params[:, 5] * inv_sqrt2
    S[:, 1, 0] = params[:, 5] * inv_sqrt2
    
    S[:, 0, 2] = params[:, 4] * inv_sqrt2
    S[:, 2, 0] = params[:, 4] * inv_sqrt2
    
    S[:, 1, 2] = params[:, 3] * inv_sqrt2
    S[:, 2, 1] = params[:, 3] * inv_sqrt2
    
    # Riemannian Exponential Map (Matrix Exponential)
    K = torch.linalg.matrix_exp(S)
    
    return K
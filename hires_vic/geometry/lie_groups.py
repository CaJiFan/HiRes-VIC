import torch

def so3_exp_map(omega: torch.Tensor) -> torch.Tensor:
        """
        Computes the Exponential Map from Lie Algebra so(3) to Lie Group SO(3).
        Input: omega (batch_size, 3) -> Axis-angle vectors (tangent space)
        Output: R (batch_size, 3, 3) -> Rotation matrices
        
        Ref: "The network provides an output s_tau in R^3... action a = Exp(s_tau)".
        """
        batch_size = omega.shape[0]
        theta = torch.norm(omega, dim=1, keepdim=True)
        epsilon = 1e-6
        
        # Handle small angles (Taylor expansion) to avoid division by zero
        # This is crucial for RL stability near convergence
        mask = theta < epsilon
        
        # Normalized axis
        u = omega / (theta + 1e-8)
        
        # Skew-symmetric matrices K
        K = torch.zeros((batch_size, 3, 3), device=omega.device)
        K[:, 0, 1] = -u[:, 2]
        K[:, 0, 2] = u[:, 1]
        K[:, 1, 0] = u[:, 2]
        K[:, 1, 2] = -u[:, 0]
        K[:, 2, 0] = -u[:, 1]
        K[:, 2, 1] = u[:, 0]
        
        # Rodrigues' Formula
        I = torch.eye(3, device=omega.device).unsqueeze(0).repeat(batch_size, 1, 1)
        R = I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta)).unsqueeze(-1) * torch.bmm(K, K)
        
        # For very small angles, R approx I + K (linear approximation)
        R[mask.squeeze()] = I[mask.squeeze()] + K[mask.squeeze()]
        
        return R

def so3_log_map(R: torch.Tensor) -> torch.Tensor:
    """
    Computes the Logarithm Map from SO(3) to so(3).
    Input: R (batch_size, 3, 3)
    Output: omega (batch_size, 3) -> Geodesic error vector
    
    Ref: "Instead of feeding orientation state s... we pass the associated Lie algebra element Log(s)".
    """
    batch_size = R.shape[0]
    epsilon = 1e-6
    
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = (tr - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + epsilon, 1 - epsilon)
    
    theta = torch.acos(cos_theta)
    
    # Standard case
    factor = theta / (2 * torch.sin(theta))
    
    omega = torch.zeros((batch_size, 3), device=R.device)
    omega[:, 0] = R[:, 2, 1] - R[:, 1, 2]
    omega[:, 1] = R[:, 0, 2] - R[:, 2, 0]
    omega[:, 2] = R[:, 1, 0] - R[:, 0, 1]
    
    omega = omega * factor.unsqueeze(-1)
    
    # Handle singularity at theta near 0
    mask = theta < epsilon
    omega[mask] = 0.5 * omega[mask] # Linear approx
    
    return omega

 
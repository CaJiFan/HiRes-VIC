import torch

def so3_exp_map(omega: torch.Tensor) -> torch.Tensor:
    """
    Computes the Exponential Map from Lie Algebra so(3) to Lie Group SO(3).
    """
    batch_size = omega.shape[0]
    theta = torch.norm(omega, dim=1, keepdim=True)
    epsilon = 1e-6
    
    # 1. Unnormalized Skew-Symmetric Matrix [omega]_x
    # (Used for the safe small-angle approximation)
    K_omega = torch.zeros((batch_size, 3, 3), device=omega.device)
    K_omega[:, 0, 1] = -omega[:, 2]
    K_omega[:, 0, 2] = omega[:, 1]
    K_omega[:, 1, 0] = omega[:, 2]
    K_omega[:, 1, 2] = -omega[:, 0]
    K_omega[:, 2, 0] = -omega[:, 1]
    K_omega[:, 2, 1] = omega[:, 0]

    # 2. Normalized Skew-Symmetric Matrix K_u
    u = omega / (theta + 1e-8)
    K_u = torch.zeros((batch_size, 3, 3), device=omega.device)
    K_u[:, 0, 1] = -u[:, 2]
    K_u[:, 0, 2] = u[:, 1]
    K_u[:, 1, 0] = u[:, 2]
    K_u[:, 1, 2] = -u[:, 0]
    K_u[:, 2, 0] = -u[:, 1]
    K_u[:, 2, 1] = u[:, 0]

    # 3. Rodrigues' Formula
    I = torch.eye(3, device=omega.device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    theta_unsqueeze = theta.unsqueeze(-1)
    R_standard = I + torch.sin(theta_unsqueeze) * K_u + (1 - torch.cos(theta_unsqueeze)) * torch.bmm(K_u, K_u)

    # 4. Safe branching for small angles: exp([w]x) approx I + [w]x
    mask = (theta < epsilon).view(-1, 1, 1)
    R = torch.where(mask, I + K_omega, R_standard)

    return R


def so3_log_map(R: torch.Tensor) -> torch.Tensor:
    """
    Computes the Logarithm Map from SO(3) to so(3).
    """
    batch_size = R.shape[0]
    epsilon = 1e-6
    
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = (tr - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + epsilon, 1 - epsilon)
    
    theta = torch.acos(cos_theta)
    
    # Extract the difference of off-diagonals
    diff = torch.zeros((batch_size, 3), device=R.device)
    diff[:, 0] = R[:, 2, 1] - R[:, 1, 2]
    diff[:, 1] = R[:, 0, 2] - R[:, 2, 0]
    diff[:, 2] = R[:, 1, 0] - R[:, 0, 1]
    
    # Safely compute the scaling factor to avoid 0/0 NaNs
    # For very small theta, theta / (2*sin(theta)) -> 0.5
    factor_standard = theta / (2 * torch.sin(theta) + 1e-8)
    factor_small = 0.5 * torch.ones_like(theta)
    
    mask = (theta < epsilon)
    factor = torch.where(mask, factor_small, factor_standard)
    
    omega = diff * factor.unsqueeze(-1)
    
    return omega
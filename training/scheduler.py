import torch


def create_scheduler(
    optimizer: torch.optim.Optimizer, max_iterations: int = 30000
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create scheduler with exponential decay ONLY for positions.
    Other parameters maintain constant learning rates.

    Args:
        optimizer: Optimizer with parameter groups in order:
                   [positions, quaternions, scales, opacities, colors]
        max_iterations: Total training iterations

    Returns:
        Configured LambdaLR scheduler
    """

    # Exponential decay for positions: lr = initial_lr * 0.01^(t/max_steps)
    def position_lr_lambda(iteration):
        return 0.01 ** (iteration / max_iterations)

    # Constant LR for all other parameters
    def constant_lr_lambda(iteration):
        return 1.0

    # One lambda per parameter group
    # Order: [positions, quaternions, scales, opacities, colors]
    lambda_functions = [
        position_lr_lambda,  # positions: decay from 0.00016 to 0.0000016
        constant_lr_lambda,  # quaternions: constant at 0.001
        constant_lr_lambda,  # scales: constant at 0.005
        constant_lr_lambda,  # opacities: constant at 0.05
        constant_lr_lambda,  # colors: constant at 0.0025
    ]

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_functions)

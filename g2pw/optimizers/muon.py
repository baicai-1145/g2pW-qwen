"""
Muon Optimizer for G2P Training
Adapted from DDSP-SVC project for G2P multi-phoneme disambiguation.

Muon - MomentUm Orthogonalized by Newton-schulz
https://kellerjordan.github.io/posts/muon/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter, Embedding
from typing import List, Dict, Any, Optional


def get_bf16_support_map():
    """Check which CUDA devices support bfloat16."""
    bf16_support_map = {}

    if not torch.cuda.is_available():
        return bf16_support_map

    device_count = torch.cuda.device_count()
    if device_count == 0:
        return bf16_support_map

    for i in range(device_count):
        device = torch.device(f'cuda:{i}')       
        major, minor = torch.cuda.get_device_capability(device)
        bf16_support_map[device] = (major >= 8)
        
    return bf16_support_map
    
    
def zeropower_via_newtonschulz5(G: Tensor, steps: int, use_bf16: bool) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    
    This quintic iteration maximizes the slope at zero for faster convergence.
    The iteration produces something like US'V^T where S' is diagonal with 
    S_{ii}' ~ Uniform(0.5, 1.5), which doesn't hurt model performance.
    """
    assert G.ndim == 3  # batched implementation
    a, b, c = (3.4445, -4.7750,  2.0315)
    
    if use_bf16:
        X = G.bfloat16()
    else:
        X = G.float()
        
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = F.normalize(X, p=2.0, dim=(-2, -1), eps=1e-7)
    
    # Perform the Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1)
    
    if G.size(-2) > G.size(-1):
        X = X.mT
        
    return X.to(G)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    Optimized for G2P tasks with large language models.
    
    Key features:
    - Orthogonalizes 2D+ parameter updates using Newton-Schulz iteration
    - Uses SGD-momentum internally with post-processing orthogonalization
    - Supports bfloat16 for memory efficiency
    - Designed for Transformer architectures
    
    Usage notes:
    - Should NOT be used for embedding layers or 1D parameters
    - Use with G2PMuonAdamW for complete optimization
    - Works well with 4D convolutional filters (flattened to 3D)
    
    Args:
        lr: Learning rate for internal SGD
        momentum: Momentum coefficient for SGD
        nesterov: Whether to use Nesterov momentum
        ns_steps: Number of Newton-Schulz iteration steps
        weight_decay: L2 regularization coefficient
    """

    def __init__(self, params, lr=5e-4, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(
            lr=lr, 
            weight_decay=weight_decay, 
            momentum=momentum, 
            nesterov=nesterov, 
            ns_steps=ns_steps
        )
        super().__init__(params, defaults)
        self.bf16_support_map = get_bf16_support_map()
        
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            # Group parameters by shape, device, and dtype for batched processing
            shape_groups = {}
            
            for p in filter(lambda p: p.grad is not None, group["params"]):
                g = p.grad
                state = self.state[p]
                
                # Initialize momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    
                buf: Tensor = state["momentum_buffer"]
                key = (p.shape, p.device, p.dtype)
                
                if key not in shape_groups:
                    shape_groups[key] = {"params": [], "grads": [], "buffers": []}
                    
                shape_groups[key]["params"].append(p)
                shape_groups[key]["grads"].append(g)
                shape_groups[key]["buffers"].append(buf)
            
            # Process each shape group
            for key in shape_groups:
                group_data = shape_groups[key]
                g = torch.stack(group_data["grads"])
                buf = torch.stack(group_data["buffers"])
                
                # Update momentum buffer
                buf.lerp_(g, 1 - group["momentum"])
                
                # Apply Nesterov momentum if enabled
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                
                # Handle convolutional filters by flattening last 3 dimensions
                if g.ndim >= 4:
                    g = g.view(g.size(0), g.size(1), -1)
                
                # Apply Newton-Schulz orthogonalization
                use_bf16 = self.bf16_support_map.get(g.device, False)
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"], use_bf16=use_bf16)
                
                # Update parameters
                for i, p in enumerate(group_data["params"]):
                    # Apply weight decay
                    if group["weight_decay"] > 0:
                        p.data.mul_(1 - group["lr"] * group["weight_decay"])
                    
                    # Apply orthogonalized update with scaling
                    scaling_factor = max(g[i].size()) ** 0.5
                    p.data.add_(g[i].view_as(p), alpha=-group["lr"] * scaling_factor)
                    
                    # Update momentum buffer
                    self.state[p]["momentum_buffer"] = buf[i].clone()
                    
        return loss


def get_params_for_muon(model) -> List[Parameter]:
    """
    Filter model parameters for Muon optimization.
    
    Returns parameters that:
    - Require gradients
    - Are 2D or higher dimensional
    - Are NOT embedding layers
    
    Args:
        model: The model to filter parameters from
        
    Returns:
        List of parameters suitable for Muon optimization
    """
    muon_params = []
    
    for module in model.modules():
        for param in module.parameters(recurse=False):
            if not param.requires_grad:
                continue
                
            # Skip embedding layers and 1D parameters
            if not isinstance(module, nn.Embedding) and param.ndim >= 2:
                muon_params.append(param)
                
    return muon_params


def get_params_for_adamw(model) -> List[Parameter]:
    """
    Filter model parameters for AdamW optimization.
    
    Returns parameters that:
    - Require gradients  
    - Are 1D parameters OR embedding layers
    
    Args:
        model: The model to filter parameters from
        
    Returns:
        List of parameters suitable for AdamW optimization
    """
    adamw_params = []
    
    for module in model.modules():
        for param in module.parameters(recurse=False):
            if not param.requires_grad:
                continue
                
            # Include embedding layers and 1D parameters
            if isinstance(module, nn.Embedding) or param.ndim < 2:
                adamw_params.append(param)
                
    return adamw_params


class G2PMuonAdamW:
    """
    Combined Muon + AdamW optimizer for G2P models.
    
    Uses Muon for 2D+ parameters (attention weights, linear layers)
    Uses AdamW for 1D parameters (biases, layer norms) and embeddings
    
    This combination is optimal for Transformer-based G2P models.
    """
    
    def __init__(self, 
                 model,
                 lr: float = 3e-5,
                 weight_decay: float = 0.01,
                 muon_momentum: float = 0.95,
                 muon_ns_steps: int = 5,
                 adamw_betas: tuple = (0.9, 0.999),
                 adamw_eps: float = 1e-8,
                 verbose: bool = False):
        """
        Initialize combined optimizer.
        
        Args:
            model: The G2P model to optimize
            lr: Learning rate for both optimizers
            weight_decay: Weight decay coefficient
            muon_momentum: Momentum for Muon optimizer
            muon_ns_steps: Newton-Schulz iteration steps
            adamw_betas: Beta coefficients for AdamW
            adamw_eps: Epsilon for AdamW
            verbose: Whether to print parameter assignment info
        """
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose
        
        # Get parameters for each optimizer
        muon_params = get_params_for_muon(model)
        adamw_params = get_params_for_adamw(model)
        
        if verbose:
            print(f"Muon parameters: {len(muon_params)}")
            print(f"AdamW parameters: {len(adamw_params)}")
            
            muon_param_count = sum(p.numel() for p in muon_params)
            adamw_param_count = sum(p.numel() for p in adamw_params)
            total_params = muon_param_count + adamw_param_count
            
            print(f"Muon parameter count: {muon_param_count:,} ({muon_param_count/total_params*100:.1f}%)")
            print(f"AdamW parameter count: {adamw_param_count:,} ({adamw_param_count/total_params*100:.1f}%)")
        
        # Initialize optimizers
        self.muon_optimizer = Muon(
            muon_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=muon_momentum,
            ns_steps=muon_ns_steps,
            nesterov=True
        )
        
        self.adamw_optimizer = torch.optim.AdamW(
            adamw_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=adamw_betas,
            eps=adamw_eps
        )
        
        self.optimizers = [self.muon_optimizer, self.adamw_optimizer]

        # Add param_groups property for compatibility
        self.param_groups = self.muon_optimizer.param_groups + self.adamw_optimizer.param_groups
    
    def step(self, closure=None):
        """Perform optimization step for both optimizers."""
        for optimizer in self.optimizers:
            optimizer.step(closure)
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients for both optimizers."""
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none)
    
    def state_dict(self):
        """Get state dict for both optimizers."""
        return {
            'muon': self.muon_optimizer.state_dict(),
            'adamw': self.adamw_optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict for both optimizers."""
        self.muon_optimizer.load_state_dict(state_dict['muon'])
        self.adamw_optimizer.load_state_dict(state_dict['adamw'])
    
    def get_lr(self):
        """Get current learning rate."""
        return self.muon_optimizer.param_groups[0]['lr']
    
    def set_lr(self, lr: float):
        """Set learning rate for both optimizers."""
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

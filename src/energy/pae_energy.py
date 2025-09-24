"""PAE Energy functions for protein structure prediction.

This module implements energy functions derived from PAE (Predicted Alignment Error) logits
using the Joint Energy-based Model (JEM) framework as described in the paper:
"PROTEIN STRUCTURE PREDICTORS IMPLICITLY DEFINE BINDING ENERGY FUNCTIONS"
"""

import torch
from torch import Tensor
from typing import Optional


def compute_pae_energy(
    pae_logits: Tensor, 
    mask: Optional[Tensor] = None
) -> Tensor:
    """Compute pAEnergy from PAE logits.
    
    pAEnergy is derived from PAE logits using the LogSumExp operation over error bins:
    E_pAE(x) = -(1/M) * Σ_{i<j} LogSumExp_y(f_pAE(x)[i,j,y])
    
    where f_pAE(x)[i,j,y] are the PAE logits for residue pair (i,j) and error bin y.
    
    Parameters
    ----------
    pae_logits : Tensor
        PAE logits tensor of shape (batch_size, num_residues, num_residues, num_bins)
        where num_bins is the number of error bins (typically 64)
    mask : Optional[Tensor], optional
        Boolean mask tensor of shape (batch_size, num_residues, num_residues) 
        indicating valid residue pairs. If None, all pairs are considered valid.
        Diagonal elements (self-pairs) are automatically excluded.
        
    Returns
    -------
    Tensor
        Scalar energy value for each batch element, shape (batch_size,)
        
    Raises
    ------
    ValueError
        If pae_logits has incorrect shape or mask has incompatible shape
    """
    # Validate input shape
    if pae_logits.dim() != 4:
        raise ValueError(
            f"pae_logits must be 4D tensor, got shape {pae_logits.shape}"
        )
    
    batch_size, num_residues, _, num_bins = pae_logits.shape
    
    # Create default mask if none provided
    if mask is None:
        mask = torch.ones(
            batch_size, num_residues, num_residues, 
            device=pae_logits.device, dtype=torch.bool
        )
    
    # Validate mask shape
    if mask.shape != (batch_size, num_residues, num_residues):
        raise ValueError(
            f"mask shape {mask.shape} must match "
            f"({batch_size}, {num_residues}, {num_residues})"
        )
    
    # Exclude diagonal elements (self-pairs) and apply mask
    diagonal_mask = ~torch.eye(
        num_residues, device=pae_logits.device, dtype=torch.bool
    ).unsqueeze(0)
    valid_pairs_mask = mask & diagonal_mask
    
    # Compute LogSumExp over error bins for each residue pair
    # Shape: (batch_size, num_residues, num_residues)
    logsumexp_pae = torch.logsumexp(pae_logits, dim=-1)
    
    # Apply mask and sum over valid residue pairs
    masked_logsumexp = logsumexp_pae * valid_pairs_mask.float()
    total_valid_pairs = valid_pairs_mask.sum(dim=(1, 2)).float()
    
    # Avoid division by zero
    total_valid_pairs = torch.clamp(total_valid_pairs, min=1.0)
    
    # Compute pAEnergy: -(1/M) * Σ_{i<j} LogSumExp_y(f_pAE(x)[i,j,y])
    pae_energy = -(masked_logsumexp.sum(dim=(1, 2)) / total_valid_pairs)
    
    return pae_energy


def compute_ptm_energy(
    pae_logits: Tensor,
    chain_ids: Tensor,
    mask: Optional[Tensor] = None,
    max_dist: float = 32.0
) -> Tensor:
    """Compute pTMEnergy from PAE logits with TM-score scaling kernel.
    
    pTMEnergy is a weighted variant of pAEnergy that incorporates the TM-score scaling kernel:
    EpTM(x) = -(1/M) * Σ_{i<j} EpTM[i,j]
    
    where:
    - EpTM[i,j] = LogSumExp_y (log g(y) + f_pAE(x)[i,j,y])
    - g(y) is the TM-score scaling kernel: g(y) = 1/(1 + y/d_0(N))^2
    - d_0(N) = 1.24 * (N-15)^(1/3) - 1.8
    - f_pAE(x)[i,j,y] are the PAE logits for residue pair (i,j) and error bin y
    - M is the number of valid inter-chain residue pairs
    
    Parameters
    ----------
    pae_logits : Tensor
        PAE logits tensor of shape (batch_size, num_residues, num_residues, num_bins)
        where num_bins is the number of error bins (typically 64)
    chain_ids : Tensor
        Chain ID tensor of shape (batch_size, num_residues) indicating which chain
        each residue belongs to
    mask : Optional[Tensor], optional
        Boolean mask tensor of shape (batch_size, num_residues, num_residues) 
        indicating valid residue pairs. If None, all pairs are considered valid.
        Diagonal elements (self-pairs) are automatically excluded.
    max_dist : float, optional
        Maximum distance for error bin scaling, by default 32.0
        
    Returns
    -------
    Tensor
        Scalar energy value for each batch element, shape (batch_size,)
        
    Raises
    ------
    ValueError
        If inputs have incorrect shapes or incompatible dimensions
    """
    # Validate input shapes
    if pae_logits.dim() != 4:
        raise ValueError(
            f"pae_logits must be 4D tensor, got shape {pae_logits.shape}"
        )
    
    batch_size, num_residues, _, num_bins = pae_logits.shape
    
    if chain_ids.shape != (batch_size, num_residues):
        raise ValueError(
            f"chain_ids shape {chain_ids.shape} must match "
            f"({batch_size}, {num_residues})"
        )
    
    # Create default mask if none provided
    if mask is None:
        mask = torch.ones(
            batch_size, num_residues, num_residues, 
            device=pae_logits.device, dtype=torch.bool
        )
    
    # Validate mask shape
    if mask.shape != (batch_size, num_residues, num_residues):
        raise ValueError(
            f"mask shape {mask.shape} must match "
            f"({batch_size}, {num_residues}, {num_residues})"
        )
    
    # Create inter-chain mask (residues from different chains)
    # Shape: (batch_size, num_residues, num_residues)
    chain_i = chain_ids.unsqueeze(-1)  # (batch_size, num_residues, 1)
    chain_j = chain_ids.unsqueeze(-2)  # (batch_size, 1, num_residues)
    inter_chain_mask = (chain_i != chain_j)  # Different chains
    
    # Exclude diagonal elements (self-pairs) and apply masks
    diagonal_mask = ~torch.eye(
        num_residues, device=pae_logits.device, dtype=torch.bool
    ).unsqueeze(0)
    valid_inter_chain_pairs = mask & diagonal_mask & inter_chain_mask
    
    # Compute TM-score scaling kernel parameters
    # d_0(N) = 1.24 * (N-15)^(1/3) - 1.8
    d0 = 1.24 * (num_residues - 15) ** (1/3) - 1.8
    
    # Create distance bins (center of each bin)
    # Assuming bins are evenly spaced from 0 to max_dist
    bin_centers = torch.linspace(
        0.0, max_dist, num_bins, device=pae_logits.device
    )
    
    # Compute TM-score scaling kernel: g(d) = 1/(1 + d/d_0)^2
    # Shape: (num_bins,)
    scaling_kernel = 1.0 / (1.0 + bin_centers / d0) ** 2
    
    # Apply scaling kernel to PAE logits according to the paper formula (10):
    # EpTM[i,j] = LogSumExp_y (log g(y) + f_pAE(x)[i,j,y])
    # Shape: (batch_size, num_residues, num_residues, num_bins)
    scaling_kernel_expanded = scaling_kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    # Add log of scaling kernel to PAE logits: log g(y) + f_pAE(x)[i,j,y]
    log_scaling_kernel = torch.log(scaling_kernel_expanded + 1e-8)  # Add small epsilon for numerical stability
    adjusted_pae_logits = pae_logits + log_scaling_kernel
    
    # Compute LogSumExp over bins: LogSumExp_y (log g(y) + f_pAE(x)[i,j,y])
    logsumexp_adjusted = torch.logsumexp(adjusted_pae_logits, dim=-1)
    
    # Apply mask and sum over valid inter-chain residue pairs
    masked_logsumexp = logsumexp_adjusted * valid_inter_chain_pairs.float()
    total_inter_chain_pairs = valid_inter_chain_pairs.sum(dim=(1, 2)).float()
    
    # Avoid division by zero
    total_inter_chain_pairs = torch.clamp(total_inter_chain_pairs, min=1.0)
    
    # Compute pTMEnergy: -(1/M) * Σ_{i<j} EpTM[i,j] where EpTM[i,j] = LogSumExp_y (log g(y) + f_pAE(x)[i,j,y])
    ptm_energy = -(masked_logsumexp.sum(dim=(1, 2)) / total_inter_chain_pairs)
    
    return ptm_energy

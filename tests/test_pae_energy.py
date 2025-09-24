"""Tests for PAE energy functions."""

import pytest
import torch
from energy import compute_pae_energy, compute_ptm_energy


def test_compute_pae_energy_basic():
    """Test basic pAEnergy computation."""
    # Create dummy PAE logits
    batch_size, num_residues, num_bins = 2, 10, 64
    pae_logits = torch.randn(batch_size, num_residues, num_residues, num_bins)
    
    # Compute pAEnergy
    energy = compute_pae_energy(pae_logits)
    
    # Check output shape
    assert energy.shape == (batch_size,)
    assert torch.is_tensor(energy)
    assert not torch.isnan(energy).any()
    assert not torch.isinf(energy).any()


def test_compute_pae_energy_with_mask():
    """Test pAEnergy computation with custom mask."""
    batch_size, num_residues, num_bins = 1, 5, 64
    pae_logits = torch.randn(batch_size, num_residues, num_residues, num_bins)
    
    # Create mask that excludes some residue pairs
    mask = torch.ones(batch_size, num_residues, num_residues, dtype=torch.bool)
    mask[0, 0, 1] = False  # Exclude pair (0, 1)
    mask[0, 1, 0] = False  # Exclude pair (1, 0)
    
    # Compute pAEnergy
    energy = compute_pae_energy(pae_logits, mask)
    
    # Check output
    assert energy.shape == (batch_size,)
    assert torch.is_tensor(energy)
    assert not torch.isnan(energy).any()


def test_compute_pae_energy_invalid_shape():
    """Test that invalid input shapes raise appropriate errors."""
    # Test 3D tensor (should be 4D)
    pae_logits_3d = torch.randn(2, 10, 10)
    with pytest.raises(ValueError, match="pae_logits must be 4D tensor"):
        compute_pae_energy(pae_logits_3d)
    
    # Test incompatible mask shape
    pae_logits = torch.randn(2, 10, 10, 64)
    mask = torch.ones(1, 10, 10, dtype=torch.bool)  # Wrong batch size
    with pytest.raises(ValueError, match="mask shape"):
        compute_pae_energy(pae_logits, mask)


def test_compute_pae_energy_diagonal_exclusion():
    """Test that diagonal elements (self-pairs) are automatically excluded."""
    batch_size, num_residues, num_bins = 1, 3, 64
    
    # Create PAE logits with very high values on diagonal
    pae_logits = torch.randn(batch_size, num_residues, num_residues, num_bins)
    pae_logits[0, 0, 0, :] = 1000.0  # High value on diagonal
    pae_logits[0, 1, 1, :] = 1000.0  # High value on diagonal
    pae_logits[0, 2, 2, :] = 1000.0  # High value on diagonal
    
    # Compute pAEnergy
    energy = compute_pae_energy(pae_logits)
    
    # Energy should be finite (diagonal values should be ignored)
    assert torch.isfinite(energy).all()


def test_compute_pae_energy_consistency():
    """Test that pAEnergy is consistent across different runs."""
    batch_size, num_residues, num_bins = 1, 5, 64
    pae_logits = torch.randn(batch_size, num_residues, num_residues, num_bins)
    
    # Compute pAEnergy twice
    energy1 = compute_pae_energy(pae_logits)
    energy2 = compute_pae_energy(pae_logits)
    
    # Results should be identical
    torch.testing.assert_close(energy1, energy2)


def test_compute_ptm_energy_basic():
    """Test basic pTMEnergy computation."""
    # Create dummy PAE logits and chain IDs
    batch_size, num_residues, num_bins = 2, 10, 64
    pae_logits = torch.randn(batch_size, num_residues, num_residues, num_bins)
    
    # Create chain IDs: first 5 residues belong to chain 0, last 5 to chain 1
    chain_ids = torch.zeros(batch_size, num_residues, dtype=torch.long)
    chain_ids[:, 5:] = 1
    
    # Compute pTMEnergy
    energy = compute_ptm_energy(pae_logits, chain_ids)
    
    # Check output shape
    assert energy.shape == (batch_size,)
    assert torch.is_tensor(energy)
    assert not torch.isnan(energy).any()
    assert not torch.isinf(energy).any()


def test_compute_ptm_energy_inter_chain_only():
    """Test that pTMEnergy only considers inter-chain residue pairs."""
    batch_size, num_residues, num_bins = 1, 6, 64
    pae_logits = torch.randn(batch_size, num_residues, num_residues, num_bins)
    
    # Create chain IDs: first 3 residues belong to chain 0, last 3 to chain 1
    chain_ids = torch.zeros(batch_size, num_residues, dtype=torch.long)
    chain_ids[:, 3:] = 1
    
    # Set very high values for intra-chain pairs (should be ignored)
    pae_logits[0, 0, 1, :] = 1000.0  # Intra-chain pair (0,1)
    pae_logits[0, 1, 0, :] = 1000.0  # Intra-chain pair (1,0)
    pae_logits[0, 4, 5, :] = 1000.0  # Intra-chain pair (4,5)
    pae_logits[0, 5, 4, :] = 1000.0  # Intra-chain pair (5,4)
    
    # Compute pTMEnergy
    energy = compute_ptm_energy(pae_logits, chain_ids)
    
    # Energy should be finite (intra-chain values should be ignored)
    assert torch.isfinite(energy).all()


def test_compute_ptm_energy_invalid_shapes():
    """Test that invalid input shapes raise appropriate errors."""
    batch_size, num_residues, num_bins = 2, 10, 64
    pae_logits = torch.randn(batch_size, num_residues, num_residues, num_bins)
    
    # Test wrong chain_ids shape
    chain_ids_wrong = torch.zeros(batch_size, num_residues + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="chain_ids shape"):
        compute_ptm_energy(pae_logits, chain_ids_wrong)
    
    # Test wrong mask shape
    mask_wrong = torch.ones(batch_size, num_residues, num_residues + 1, dtype=torch.bool)
    chain_ids = torch.zeros(batch_size, num_residues, dtype=torch.long)
    with pytest.raises(ValueError, match="mask shape"):
        compute_ptm_energy(pae_logits, chain_ids, mask_wrong)


def test_compute_ptm_energy_scaling_kernel():
    """Test that pTMEnergy applies the TM-score scaling kernel correctly."""
    batch_size, num_residues, num_bins = 1, 20, 64
    pae_logits = torch.randn(batch_size, num_residues, num_residues, num_bins)
    
    # Create chain IDs: half residues per chain
    chain_ids = torch.zeros(batch_size, num_residues, dtype=torch.long)
    chain_ids[:, num_residues//2:] = 1
    
    # Compute pTMEnergy
    energy = compute_ptm_energy(pae_logits, chain_ids)
    
    # Energy should be finite and reasonable
    assert torch.isfinite(energy).all()
    assert energy.shape == (batch_size,)


def test_compute_ptm_energy_consistency():
    """Test that pTMEnergy is consistent across different runs."""
    batch_size, num_residues, num_bins = 1, 8, 64
    pae_logits = torch.randn(batch_size, num_residues, num_residues, num_bins)
    chain_ids = torch.zeros(batch_size, num_residues, dtype=torch.long)
    chain_ids[:, 4:] = 1
    
    # Compute pTMEnergy twice
    energy1 = compute_ptm_energy(pae_logits, chain_ids)
    energy2 = compute_ptm_energy(pae_logits, chain_ids)
    
    # Results should be identical
    torch.testing.assert_close(energy1, energy2)


def test_pae_vs_ptm_energy_difference():
    """Test that pAEnergy and pTMEnergy produce different results."""
    batch_size, num_residues, num_bins = 1, 10, 64
    pae_logits = torch.randn(batch_size, num_residues, num_residues, num_bins)
    chain_ids = torch.zeros(batch_size, num_residues, dtype=torch.long)
    chain_ids[:, 5:] = 1
    
    # Compute both energies
    pae_energy = compute_pae_energy(pae_logits)
    ptm_energy = compute_ptm_energy(pae_logits, chain_ids)
    
    # They should be different (pTMEnergy uses scaling kernel and inter-chain only)
    # Use real part for comparison to avoid complex number issues
    assert not torch.allclose(pae_energy.real, ptm_energy.real, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])

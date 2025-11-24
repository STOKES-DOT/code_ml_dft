#!/usr/bin/env python3
"""
Davidson Diagonalization Comparison Script
==========================================
This script generates a random Hermitian matrix, diagonalizes it using both
standard methods and Davidson diagonalization, then compares the results.

Author: Generated for OPTXC project
"""

import numpy as np
from scipy.sparse.linalg import eigsh
import time


def generate_hermitian_matrix(size, density=0.5, seed=None):
    """
    Generate a random Hermitian matrix.
    
    Parameters:
    -----------
    size : int
        Size of the matrix (size x size)
    density : float
        Density of non-zero elements (0 to 1)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    numpy.ndarray
        A Hermitian matrix of shape (size, size)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random complex matrix
    A = np.random.randn(size, size) + 1j * np.random.randn(size, size)
    
    # Apply sparsity
    mask = np.random.rand(size, size) < density
    A = A * mask
    
    # Make it Hermitian: H = (A + A†) / 2
    H = (A + A.conj().T) / 2
    
    return H


def standard_diagonalization(H, num_eigenvalues=3):
    """
    Perform standard full diagonalization using numpy/scipy.
    
    Parameters:
    -----------
    H : numpy.ndarray
        Hermitian matrix to diagonalize
    num_eigenvalues : int
        Number of lowest eigenvalues to return
        
    Returns:
    --------
    eigenvalues : numpy.ndarray
        The lowest eigenvalues
    eigenvectors : numpy.ndarray
        Corresponding eigenvectors
    elapsed_time : float
        Time taken for diagonalization
    """
    start_time = time.time()
    
    # For smaller matrices, use numpy's eigh
    if H.shape[0] <= 1000:
        eigenvalues, eigenvectors = np.linalg.eigh(H)
    else:
        # For larger matrices, use sparse methods
        eigenvalues, eigenvectors = eigsh(H, k=num_eigenvalues, which='SA')
    
    elapsed_time = time.time() - start_time
    
    # Return only the requested number of eigenvalues
    return eigenvalues[:num_eigenvalues], eigenvectors[:, :num_eigenvalues], elapsed_time


def davidson_diagonalization(H, num_eigenvalues=3, max_iterations=1000, tolerance=1e-8, seed=None):
    """
    Perform Davidson diagonalization to find the lowest eigenvalues.
    
    The Davidson algorithm is an iterative method particularly efficient for 
    finding a few eigenvalues of large sparse Hermitian matrices.
    
    Parameters:
    -----------
    H : numpy.ndarray
        Hermitian matrix to diagonalize
    num_eigenvalues : int
        Number of lowest eigenvalues to find
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance
    seed : int, optional
        Random seed for reproducibility of initial guess
        
    Returns:
    --------
    eigenvalues : numpy.ndarray
        Approximated lowest eigenvalues
    eigenvectors : numpy.ndarray
        Approximated corresponding eigenvectors
    elapsed_time : float
        Time taken for diagonalization
    iterations : int
        Number of iterations performed
    """
    start_time = time.time()
    
    n = H.shape[0]
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with random guess vectors
    V = np.random.randn(n, num_eigenvalues) + 1j * np.random.randn(n, num_eigenvalues)
    
    # Orthonormalize initial vectors
    V, _ = np.linalg.qr(V)
    
    eigenvalues = np.zeros(num_eigenvalues, dtype=complex)
    eigenvectors = np.zeros((n, num_eigenvalues), dtype=complex)
    
    # Extract diagonal once for use in preconditioner
    diag_H = np.diag(H)
    
    for iteration in range(max_iterations):
        # Project H onto the subspace spanned by V
        HV = H @ V
        T = V.conj().T @ HV  # Small projected matrix
        
        # Diagonalize the small projected matrix
        theta, s = np.linalg.eigh(T)
        
        # Compute Ritz vectors
        ritz_vectors = V @ s
        
        # Compute residuals
        residuals = HV @ s - ritz_vectors @ np.diag(theta)
        residual_norms = np.linalg.norm(residuals, axis=0)
        
        # Check convergence
        if np.max(residual_norms[:num_eigenvalues]) < tolerance:
            eigenvalues = theta[:num_eigenvalues]
            eigenvectors = ritz_vectors[:, :num_eigenvalues]
            break
        
        # Prepare correction vectors
        corrections = np.zeros((n, num_eigenvalues), dtype=complex)
        
        for i in range(num_eigenvalues):
            # Simple diagonal preconditioner
            denominator = diag_H - theta[i]
            # Avoid division by zero
            denominator[np.abs(denominator) < 1e-10] = 1e-10
            corrections[:, i] = -residuals[:, i] / denominator
        
        # Expand the subspace
        V_new = np.hstack([V, corrections])
        
        # Orthonormalize the expanded subspace
        V, _ = np.linalg.qr(V_new)
        
        # Limit the size of the subspace
        if V.shape[1] > num_eigenvalues * 10:
            # Restart with best vectors
            V = ritz_vectors[:, :num_eigenvalues * 2]
            V, _ = np.linalg.qr(V)
    
    elapsed_time = time.time() - start_time
    
    # Convert to real if imaginary part is negligible
    if np.max(np.abs(eigenvalues.imag)) < 1e-10:
        eigenvalues = eigenvalues.real
    
    return eigenvalues, eigenvectors, elapsed_time, iteration + 1


def compare_results(standard_evals, standard_evecs, davidson_evals, davidson_evecs):
    """
    Compare the results from standard and Davidson diagonalization.
    
    Parameters:
    -----------
    standard_evals : numpy.ndarray
        Eigenvalues from standard diagonalization
    standard_evecs : numpy.ndarray
        Eigenvectors from standard diagonalization
    davidson_evals : numpy.ndarray
        Eigenvalues from Davidson diagonalization
    davidson_evecs : numpy.ndarray
        Eigenvectors from Davidson diagonalization
        
    Returns:
    --------
    dict
        Dictionary containing comparison metrics
    """
    comparison = {}
    
    # Eigenvalue differences
    eigenvalue_diff = np.abs(standard_evals - davidson_evals)
    comparison['eigenvalue_diff'] = eigenvalue_diff
    comparison['max_eigenvalue_diff'] = np.max(eigenvalue_diff)
    comparison['mean_eigenvalue_diff'] = np.mean(eigenvalue_diff)
    
    # Eigenvector overlap (should be close to 1 for matching eigenvectors)
    overlaps = []
    for i in range(standard_evecs.shape[1]):
        overlap = np.abs(np.vdot(standard_evecs[:, i], davidson_evecs[:, i]))
        overlaps.append(overlap)
    
    comparison['eigenvector_overlaps'] = np.array(overlaps)
    comparison['min_overlap'] = np.min(overlaps)
    comparison['mean_overlap'] = np.mean(overlaps)
    
    return comparison


def print_results(H_size, standard_evals, standard_time, davidson_evals, davidson_time, 
                  davidson_iters, comparison):
    """
    Print formatted comparison results.
    """
    print("\n" + "="*70)
    print(f"HERMITIAN MATRIX DIAGONALIZATION COMPARISON")
    print("="*70)
    print(f"\nMatrix Size: {H_size} × {H_size}")
    print(f"\n{'Method':<20} {'Time (s)':<15} {'Iterations':<15}")
    print("-"*70)
    print(f"{'Standard':<20} {standard_time:<15.6f} {'N/A':<15}")
    print(f"{'Davidson':<20} {davidson_time:<15.6f} {davidson_iters:<15}")
    
    print(f"\n{'Eigenstate':<15} {'Standard':<20} {'Davidson':<20} {'Difference':<20}")
    print("-"*70)
    for i in range(len(standard_evals)):
        print(f"{i+1:<15} {standard_evals[i]:<20.10f} {davidson_evals[i]:<20.10f} "
              f"{comparison['eigenvalue_diff'][i]:<20.2e}")
    
    print(f"\nEigenvalue Comparison:")
    print(f"  - Maximum difference: {comparison['max_eigenvalue_diff']:.2e}")
    print(f"  - Mean difference:    {comparison['mean_eigenvalue_diff']:.2e}")
    
    print(f"\nEigenvector Overlap (1.0 = perfect match):")
    for i, overlap in enumerate(comparison['eigenvector_overlaps']):
        print(f"  - Eigenstate {i+1}: {overlap:.10f}")
    print(f"  - Minimum overlap:  {comparison['min_overlap']:.10f}")
    print(f"  - Mean overlap:     {comparison['mean_overlap']:.10f}")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    if comparison['max_eigenvalue_diff'] < 1e-6 and comparison['min_overlap'] > 0.9999:
        print("✓ Excellent agreement between standard and Davidson methods!")
    elif comparison['max_eigenvalue_diff'] < 1e-4 and comparison['min_overlap'] > 0.999:
        print("✓ Good agreement between standard and Davidson methods.")
    else:
        print("⚠ Some discrepancy observed. Consider adjusting Davidson parameters.")
    print("="*70 + "\n")


def main():
    """
    Main function to run the comparison.
    """
    # Parameters
    matrix_size = 100  # Size of the Hermitian matrix
    num_eigenvalues = 3  # Number of lowest eigenstates to find
    random_seed = 42  # For reproducibility
    
    print("\n" + "="*70)
    print("HERMITIAN MATRIX DIAGONALIZATION - DAVIDSON vs STANDARD METHODS")
    print("="*70)
    
    # Generate random Hermitian matrix
    print(f"\nGenerating {matrix_size}×{matrix_size} Hermitian matrix...")
    H = generate_hermitian_matrix(matrix_size, density=0.5, seed=random_seed)
    print(f"✓ Matrix generated (Hermitian property verified: {np.allclose(H, H.conj().T)})")
    
    # Standard diagonalization
    print(f"\nPerforming standard diagonalization...")
    standard_evals, standard_evecs, standard_time = standard_diagonalization(H, num_eigenvalues)
    print(f"✓ Standard diagonalization completed in {standard_time:.6f} seconds")
    
    # Davidson diagonalization
    print(f"\nPerforming Davidson diagonalization...")
    davidson_evals, davidson_evecs, davidson_time, davidson_iters = davidson_diagonalization(
        H, num_eigenvalues, max_iterations=1000, tolerance=1e-8
    )
    print(f"✓ Davidson diagonalization completed in {davidson_time:.6f} seconds ({davidson_iters} iterations)")
    
    # Compare results
    comparison = compare_results(standard_evals, standard_evecs, davidson_evals, davidson_evecs)
    
    # Print results
    print_results(matrix_size, standard_evals, standard_time, davidson_evals, 
                  davidson_time, davidson_iters, comparison)


if __name__ == "__main__":
    main()

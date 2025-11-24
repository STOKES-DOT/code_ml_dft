#!/usr/bin/env python3
"""
Davidson Diagonalization Example
=================================
A simple example demonstrating how to use the Davidson diagonalization
comparison tools.

Usage:
    python example_davidson.py              # Run with default parameters
    python example_davidson.py 150          # Custom matrix size
    python example_davidson.py 150 5        # Custom size and num_eigenvalues

Author: Generated for OPTXC project
"""

import sys
import numpy as np
from davidson_comparison import (
    generate_hermitian_matrix,
    standard_diagonalization,
    davidson_diagonalization,
    compare_results,
    print_results
)


def example_basic():
    """
    Basic example: Compare standard and Davidson diagonalization.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Comparison")
    print("="*70)
    
    # Generate a Hermitian matrix
    size = 80
    num_evals = 3
    
    print(f"\nGenerating {size}×{size} Hermitian matrix...")
    H = generate_hermitian_matrix(size, density=0.5, seed=42)
    
    # Verify it's Hermitian
    is_hermitian = np.allclose(H, H.conj().T)
    print(f"Matrix is Hermitian: {is_hermitian}")
    
    # Standard diagonalization
    print(f"\nComputing lowest {num_evals} eigenvalues using standard method...")
    std_evals, std_evecs, std_time = standard_diagonalization(H, num_evals)
    
    # Davidson diagonalization
    print(f"Computing lowest {num_evals} eigenvalues using Davidson method...")
    dav_evals, dav_evecs, dav_time, dav_iters = davidson_diagonalization(
        H, num_evals, max_iterations=1000, tolerance=1e-8
    )
    
    # Compare results
    comparison = compare_results(std_evals, std_evecs, dav_evals, dav_evecs)
    
    # Print detailed results
    print_results(size, std_evals, std_time, dav_evals, dav_time, 
                  dav_iters, comparison)


def example_custom(size=100, num_eigenvalues=3):
    """
    Custom example with user-specified parameters.
    
    Parameters:
    -----------
    size : int
        Size of the matrix
    num_eigenvalues : int
        Number of lowest eigenvalues to find
    """
    print("\n" + "="*70)
    print(f"EXAMPLE 2: Custom Parameters (size={size}, k={num_eigenvalues})")
    print("="*70)
    
    print(f"\nGenerating {size}×{size} Hermitian matrix...")
    H = generate_hermitian_matrix(size, density=0.6, seed=123)
    
    print("\nMatrix properties:")
    print(f"  - Shape: {H.shape}")
    print(f"  - Hermitian: {np.allclose(H, H.conj().T)}")
    print(f"  - Density: ~60%")
    
    # Standard method
    print(f"\nStandard diagonalization...")
    std_evals, std_evecs, std_time = standard_diagonalization(H, num_eigenvalues)
    print(f"  Time: {std_time:.6f}s")
    print(f"  Eigenvalues: {std_evals}")
    
    # Davidson method
    print(f"\nDavidson diagonalization...")
    dav_evals, dav_evecs, dav_time, dav_iters = davidson_diagonalization(
        H, num_eigenvalues, max_iterations=1000, tolerance=1e-8
    )
    print(f"  Time: {dav_time:.6f}s")
    print(f"  Iterations: {dav_iters}")
    print(f"  Eigenvalues: {dav_evals}")
    
    # Comparison
    comparison = compare_results(std_evals, std_evecs, dav_evals, dav_evecs)
    
    print(f"\nComparison:")
    print(f"  - Max eigenvalue difference: {comparison['max_eigenvalue_diff']:.2e}")
    print(f"  - Mean eigenvector overlap: {comparison['mean_overlap']:.10f}")
    
    if comparison['max_eigenvalue_diff'] < 1e-6:
        print("\n✓ Excellent numerical agreement!")
    else:
        print("\n⚠ Some numerical discrepancy detected.")
    
    print("="*70 + "\n")


def example_convergence_study():
    """
    Example showing how tolerance affects Davidson convergence.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Convergence Study")
    print("="*70)
    
    size = 100
    num_evals = 3
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    
    print(f"\nGenerating {size}×{size} Hermitian matrix...")
    H = generate_hermitian_matrix(size, density=0.5, seed=42)
    
    # Get reference solution
    ref_evals, _, _ = standard_diagonalization(H, num_evals)
    
    print(f"\nTesting different tolerance values:")
    print(f"{'Tolerance':<15} {'Iterations':<12} {'Time (s)':<12} {'Max Error':<15}")
    print("-"*70)
    
    for tol in tolerances:
        dav_evals, _, dav_time, dav_iters = davidson_diagonalization(
            H, num_evals, max_iterations=1000, tolerance=tol
        )
        max_error = np.max(np.abs(ref_evals - dav_evals))
        print(f"{tol:<15.0e} {dav_iters:<12} {dav_time:<12.6f} {max_error:<15.2e}")
    
    print("\nObservation: Tighter tolerance requires more iterations but gives")
    print("             higher accuracy. Choose based on application needs.")
    print("="*70 + "\n")


def example_large_matrix():
    """
    Example demonstrating efficiency for larger matrices.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Large Matrix Efficiency")
    print("="*70)
    
    size = 500
    num_evals = 3
    
    print(f"\nNote: For large matrices ({size}×{size}), Davidson method becomes")
    print("      more competitive, especially when only few eigenvalues are needed.\n")
    
    print(f"Generating {size}×{size} Hermitian matrix...")
    H = generate_hermitian_matrix(size, density=0.3, seed=999)
    
    print("\nComputing 3 lowest eigenvalues...")
    
    # Standard method (using sparse eigsh for fairness)
    print("  Standard method (sparse)...")
    std_evals, std_evecs, std_time = standard_diagonalization(H, num_evals)
    print(f"    Time: {std_time:.6f}s")
    
    # Davidson method
    print("  Davidson method...")
    dav_evals, dav_evecs, dav_time, dav_iters = davidson_diagonalization(
        H, num_evals, max_iterations=2000, tolerance=1e-8
    )
    print(f"    Time: {dav_time:.6f}s")
    print(f"    Iterations: {dav_iters}")
    
    # Compare
    comparison = compare_results(std_evals, std_evecs, dav_evals, dav_evecs)
    
    print(f"\nResults:")
    print(f"  Eigenvalues match within: {comparison['max_eigenvalue_diff']:.2e}")
    print(f"  Eigenvector overlap: {comparison['mean_overlap']:.10f}")
    
    speedup = std_time / dav_time
    if speedup > 1:
        print(f"  Davidson is {speedup:.2f}x faster!")
    else:
        print(f"  Standard method is {1/speedup:.2f}x faster.")
    
    print("="*70 + "\n")


def main():
    """
    Main function to run examples.
    """
    print("\n" + "="*70)
    print(" "*15 + "DAVIDSON DIAGONALIZATION EXAMPLES")
    print("="*70)
    print("\nThese examples demonstrate how to use Davidson diagonalization")
    print("to find eigenvalues of Hermitian matrices.\n")
    
    if len(sys.argv) > 1:
        # Custom parameters from command line
        try:
            size = int(sys.argv[1])
            num_evals = int(sys.argv[2]) if len(sys.argv) > 2 else 3
            example_custom(size, num_evals)
        except ValueError:
            print("Error: Invalid parameters. Using defaults.")
            example_basic()
    else:
        # Run all examples
        example_basic()
        
        # Uncomment to run additional examples
        # example_custom(120, 4)
        # example_convergence_study()
        # example_large_matrix()
    
    print("\n" + "="*70)
    print("For more examples, see davidson_comparison.py and davidson_benchmark.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

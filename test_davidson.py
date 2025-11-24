#!/usr/bin/env python3
"""
Unit tests for Davidson diagonalization implementation.
"""

import numpy as np
from davidson_comparison import (
    generate_hermitian_matrix,
    standard_diagonalization,
    davidson_diagonalization,
    compare_results
)


def test_hermitian_generation():
    """Test that generated matrices are Hermitian."""
    print("\n" + "="*70)
    print("TEST 1: Hermitian Matrix Generation")
    print("="*70)
    
    sizes = [10, 50, 100]
    for size in sizes:
        H = generate_hermitian_matrix(size, density=0.5, seed=42)
        is_hermitian = np.allclose(H, H.conj().T)
        print(f"  Size {size}×{size}: {'✓ PASS' if is_hermitian else '✗ FAIL'}")
        assert is_hermitian, f"Matrix of size {size} is not Hermitian!"
    
    print("✓ All matrices are Hermitian\n")


def test_eigenvalue_accuracy():
    """Test that Davidson produces accurate eigenvalues."""
    print("="*70)
    print("TEST 2: Eigenvalue Accuracy")
    print("="*70)
    
    size = 80
    num_evals = 3
    H = generate_hermitian_matrix(size, density=0.5, seed=123)
    
    std_evals, _, _ = standard_diagonalization(H, num_evals)
    dav_evals, _, _, _ = davidson_diagonalization(H, num_evals, tolerance=1e-8)
    
    max_diff = np.max(np.abs(std_evals - dav_evals))
    print(f"  Matrix size: {size}×{size}")
    print(f"  Num eigenvalues: {num_evals}")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Status: {'✓ PASS' if max_diff < 1e-6 else '✗ FAIL'}")
    
    assert max_diff < 1e-6, f"Eigenvalue difference too large: {max_diff}"
    print("✓ Eigenvalues match within tolerance\n")


def test_eigenvector_orthogonality():
    """Test that eigenvectors are orthonormal."""
    print("="*70)
    print("TEST 3: Eigenvector Orthonormality")
    print("="*70)
    
    size = 60
    num_evals = 3
    H = generate_hermitian_matrix(size, density=0.5, seed=456)
    
    _, dav_evecs, _, _ = davidson_diagonalization(H, num_evals, tolerance=1e-8)
    
    # Check orthonormality
    gram = dav_evecs.conj().T @ dav_evecs
    identity = np.eye(num_evals)
    
    max_deviation = np.max(np.abs(gram - identity))
    print(f"  Matrix size: {size}×{size}")
    print(f"  Num eigenvectors: {num_evals}")
    print(f"  Max deviation from orthonormality: {max_deviation:.2e}")
    print(f"  Status: {'✓ PASS' if max_deviation < 1e-6 else '✗ FAIL'}")
    
    assert max_deviation < 1e-6, f"Eigenvectors not orthonormal: {max_deviation}"
    print("✓ Eigenvectors are orthonormal\n")


def test_convergence():
    """Test that Davidson converges for different tolerances."""
    print("="*70)
    print("TEST 4: Convergence with Different Tolerances")
    print("="*70)
    
    size = 70
    num_evals = 3
    H = generate_hermitian_matrix(size, density=0.5, seed=789)
    
    tolerances = [1e-4, 1e-6, 1e-8]
    std_evals, _, _ = standard_diagonalization(H, num_evals)
    
    print(f"  Matrix size: {size}×{size}")
    print(f"  {'Tolerance':<12} {'Iterations':<12} {'Max Error':<15} {'Status':<10}")
    print("  " + "-"*60)
    
    for tol in tolerances:
        dav_evals, _, _, iters = davidson_diagonalization(
            H, num_evals, tolerance=tol, max_iterations=1000
        )
        max_error = np.max(np.abs(std_evals - dav_evals))
        status = "✓ PASS" if max_error < tol * 100 else "✗ FAIL"
        print(f"  {tol:<12.0e} {iters:<12} {max_error:<15.2e} {status:<10}")
        assert max_error < tol * 100, f"Did not converge for tolerance {tol}"
    
    print("✓ Davidson converges for all tolerances\n")


def test_eigenvalue_verification():
    """Verify Hv = λv for computed eigenpairs."""
    print("="*70)
    print("TEST 5: Eigenvalue Equation Verification")
    print("="*70)
    
    size = 50
    num_evals = 3
    H = generate_hermitian_matrix(size, density=0.5, seed=101)
    
    dav_evals, dav_evecs, _, _ = davidson_diagonalization(H, num_evals, tolerance=1e-8)
    
    print(f"  Matrix size: {size}×{size}")
    print(f"  {'Eigenvalue':<12} {'||Hv - λv||':<15} {'Status':<10}")
    print("  " + "-"*45)
    
    for i in range(num_evals):
        v = dav_evecs[:, i]
        lam = dav_evals[i]
        
        Hv = H @ v
        lam_v = lam * v
        residual = np.linalg.norm(Hv - lam_v)
        status = "✓ PASS" if residual < 1e-6 else "✗ FAIL"
        
        print(f"  {lam:<12.6f} {residual:<15.2e} {status:<10}")
        assert residual < 1e-6, f"Eigenvalue equation not satisfied: {residual}"
    
    print("✓ All eigenpairs satisfy Hv = λv\n")


def test_different_sizes():
    """Test Davidson on different matrix sizes."""
    print("="*70)
    print("TEST 6: Different Matrix Sizes")
    print("="*70)
    
    sizes = [30, 60, 100, 150]
    num_evals = 3
    
    print(f"  {'Size':<10} {'Max Diff':<15} {'Min Overlap':<15} {'Status':<10}")
    print("  " + "-"*55)
    
    for size in sizes:
        H = generate_hermitian_matrix(size, density=0.5, seed=size)
        
        std_evals, std_evecs, _ = standard_diagonalization(H, num_evals)
        dav_evals, dav_evecs, _, _ = davidson_diagonalization(H, num_evals, tolerance=1e-8)
        
        comparison = compare_results(std_evals, std_evecs, dav_evals, dav_evecs)
        max_diff = comparison['max_eigenvalue_diff']
        min_overlap = comparison['min_overlap']
        
        status = "✓ PASS" if max_diff < 1e-6 and min_overlap > 0.9999 else "✗ FAIL"
        print(f"  {size:<10} {max_diff:<15.2e} {min_overlap:<15.10f} {status:<10}")
        
        assert max_diff < 1e-6, f"Failed for size {size}"
        assert min_overlap > 0.9999, f"Poor overlap for size {size}"
    
    print("✓ Davidson works for all tested sizes\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*20 + "DAVIDSON DIAGONALIZATION TESTS")
    print("="*70)
    print("\nRunning comprehensive test suite...\n")
    
    tests = [
        test_hermitian_generation,
        test_eigenvalue_accuracy,
        test_eigenvector_orthogonality,
        test_convergence,
        test_eigenvalue_verification,
        test_different_sizes
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ TEST FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ TEST ERROR: {e}\n")
            failed += 1
    
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n✗ {failed} TEST(S) FAILED!")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Davidson Diagonalization Interactive Demo
==========================================
A comprehensive demonstration showing various aspects of Davidson diagonalization.

Author: Generated for OPTXC project
"""

import numpy as np
from davidson_comparison import (
    generate_hermitian_matrix,
    standard_diagonalization,
    davidson_diagonalization,
    compare_results
)


def demo_basic_comparison():
    """
    Demo 1: Basic comparison between standard and Davidson methods.
    """
    print("\n" + "="*80)
    print(" "*25 + "DEMO 1: BASIC COMPARISON")
    print("="*80)
    
    print("\n本示例演示如何生成厄米矩阵并使用两种方法对角化")
    print("This demo shows how to generate a Hermitian matrix and diagonalize it")
    print("using both standard and Davidson methods.\n")
    
    # Parameters
    size = 100
    num_evals = 3
    
    # Step 1: Generate Hermitian matrix
    print(f"步骤 1: 生成 {size}×{size} 厄米矩阵")
    print(f"Step 1: Generate a {size}×{size} Hermitian matrix")
    H = generate_hermitian_matrix(size, density=0.5, seed=42)
    print(f"  ✓ 矩阵已生成 (Matrix generated)")
    print(f"  - 厄米性验证 (Hermitian check): {np.allclose(H, H.conj().T)}")
    print(f"  - 非零元素 (Non-zero elements): {np.count_nonzero(H)} / {size*size}")
    
    # Step 2: Standard diagonalization
    print(f"\n步骤 2: 标准对角化 (计算最小的 {num_evals} 个本征值)")
    print(f"Step 2: Standard diagonalization (computing lowest {num_evals} eigenvalues)")
    std_evals, std_evecs, std_time = standard_diagonalization(H, num_evals)
    print(f"  ✓ 完成 (Completed)")
    print(f"  - 时间 (Time): {std_time:.6f} 秒 (seconds)")
    print(f"  - 本征值 (Eigenvalues):")
    for i, ev in enumerate(std_evals):
        print(f"    λ_{i+1} = {ev:.10f}")
    
    # Step 3: Davidson diagonalization
    print(f"\n步骤 3: Davidson对角化 (计算最小的 {num_evals} 个本征值)")
    print(f"Step 3: Davidson diagonalization (computing lowest {num_evals} eigenvalues)")
    dav_evals, dav_evecs, dav_time, dav_iters = davidson_diagonalization(
        H, num_evals, max_iterations=1000, tolerance=1e-8
    )
    print(f"  ✓ 完成 (Completed)")
    print(f"  - 时间 (Time): {dav_time:.6f} 秒 (seconds)")
    print(f"  - 迭代次数 (Iterations): {dav_iters}")
    print(f"  - 本征值 (Eigenvalues):")
    for i, ev in enumerate(dav_evals):
        print(f"    λ_{i+1} = {ev:.10f}")
    
    # Step 4: Compare results
    print(f"\n步骤 4: 结果对比")
    print(f"Step 4: Results comparison")
    comparison = compare_results(std_evals, std_evecs, dav_evals, dav_evecs)
    
    print(f"\n  本征值差异 (Eigenvalue differences):")
    for i in range(num_evals):
        diff = abs(std_evals[i] - dav_evals[i])
        print(f"    |λ_{i+1}^std - λ_{i+1}^dav| = {diff:.2e}")
    
    print(f"\n  本征向量重叠 (Eigenvector overlaps):")
    for i, overlap in enumerate(comparison['eigenvector_overlaps']):
        print(f"    |⟨v_{i+1}^std | v_{i+1}^dav⟩| = {overlap:.10f}")
    
    print(f"\n总结 (Summary):")
    print(f"  - 最大本征值差异 (Max eigenvalue diff): {comparison['max_eigenvalue_diff']:.2e}")
    print(f"  - 最小本征向量重叠 (Min eigenvector overlap): {comparison['min_overlap']:.10f}")
    
    if comparison['max_eigenvalue_diff'] < 1e-10:
        print(f"  ✓ 两种方法结果完全一致！ (Excellent agreement!)")
    
    print("\n" + "="*80 + "\n")


def demo_matrix_properties():
    """
    Demo 2: Show properties of generated Hermitian matrices.
    """
    print("="*80)
    print(" "*20 + "DEMO 2: HERMITIAN MATRIX PROPERTIES")
    print("="*80)
    
    print("\n本示例展示生成的厄米矩阵的性质")
    print("This demo shows properties of generated Hermitian matrices.\n")
    
    size = 50
    H = generate_hermitian_matrix(size, density=0.5, seed=42)
    
    print(f"矩阵大小 (Matrix size): {size}×{size}")
    
    # Property 1: Hermitian
    print(f"\n性质 1: 厄米性 (Property 1: Hermitian)")
    is_hermitian = np.allclose(H, H.conj().T)
    print(f"  H = H† ? {is_hermitian}")
    if is_hermitian:
        print(f"  ✓ 矩阵是厄米的 (Matrix is Hermitian)")
    
    # Property 2: Real eigenvalues
    print(f"\n性质 2: 实数本征值 (Property 2: Real eigenvalues)")
    evals = np.linalg.eigvalsh(H)
    print(f"  前5个本征值 (First 5 eigenvalues):")
    for i in range(min(5, len(evals))):
        print(f"    λ_{i+1} = {evals[i]:.10f}")
    print(f"  ✓ 所有本征值都是实数 (All eigenvalues are real)")
    
    # Property 3: Eigenvalue spectrum
    print(f"\n性质 3: 本征值谱 (Property 3: Eigenvalue spectrum)")
    print(f"  最小本征值 (Min eigenvalue): {np.min(evals):.6f}")
    print(f"  最大本征值 (Max eigenvalue): {np.max(evals):.6f}")
    print(f"  本征值范围 (Eigenvalue range): {np.max(evals) - np.min(evals):.6f}")
    print(f"  平均本征值 (Mean eigenvalue): {np.mean(evals):.6f}")
    
    # Property 4: Matrix norms
    print(f"\n性质 4: 矩阵范数 (Property 4: Matrix norms)")
    frobenius = np.linalg.norm(H, 'fro')
    spectral = np.max(np.abs(evals))
    print(f"  Frobenius范数 (Frobenius norm): {frobenius:.6f}")
    print(f"  谱范数 (Spectral norm): {spectral:.6f}")
    
    print("\n" + "="*80 + "\n")


def demo_convergence_analysis():
    """
    Demo 3: Analyze Davidson convergence behavior.
    """
    print("="*80)
    print(" "*23 + "DEMO 3: CONVERGENCE ANALYSIS")
    print("="*80)
    
    print("\n本示例分析Davidson算法的收敛行为")
    print("This demo analyzes the convergence behavior of Davidson algorithm.\n")
    
    size = 80
    num_evals = 3
    H = generate_hermitian_matrix(size, density=0.5, seed=42)
    
    # Get reference solution
    ref_evals, _, _ = standard_diagonalization(H, num_evals)
    
    print(f"矩阵大小 (Matrix size): {size}×{size}")
    print(f"计算本征值数量 (Number of eigenvalues): {num_evals}")
    print(f"\n测试不同的容差值 (Testing different tolerance values):")
    print(f"\n{'容差 Tolerance':<18} {'迭代 Iters':<12} {'时间 Time (s)':<15} "
          f"{'最大误差 Max Error':<20}")
    print("-"*80)
    
    tolerances = [1e-3, 1e-4, 1e-6, 1e-8, 1e-10]
    
    for tol in tolerances:
        dav_evals, _, dav_time, dav_iters = davidson_diagonalization(
            H, num_evals, max_iterations=2000, tolerance=tol
        )
        max_error = np.max(np.abs(ref_evals - dav_evals))
        print(f"{tol:<18.0e} {dav_iters:<12} {dav_time:<15.6f} {max_error:<20.2e}")
    
    print(f"\n观察 (Observations):")
    print(f"  1. 更严格的容差需要更多迭代 (Tighter tolerance requires more iterations)")
    print(f"  2. 但能得到更高的精度 (But achieves higher accuracy)")
    print(f"  3. 推荐使用 1e-8 以获得良好的平衡 (Recommend 1e-8 for good balance)")
    
    print("\n" + "="*80 + "\n")


def demo_size_scaling():
    """
    Demo 4: Show how performance scales with matrix size.
    """
    print("="*80)
    print(" "*25 + "DEMO 4: SIZE SCALING")
    print("="*80)
    
    print("\n本示例展示性能如何随矩阵大小变化")
    print("This demo shows how performance scales with matrix size.\n")
    
    sizes = [30, 50, 80, 100, 150]
    num_evals = 3
    
    print(f"计算本征值数量 (Number of eigenvalues): {num_evals}")
    print(f"\n{'矩阵大小 Size':<15} {'标准方法 Std (s)':<18} {'Davidson (s)':<18} "
          f"{'加速比 Speedup':<15}")
    print("-"*80)
    
    for size in sizes:
        H = generate_hermitian_matrix(size, density=0.5, seed=size)
        
        _, _, std_time = standard_diagonalization(H, num_evals)
        _, _, dav_time, _ = davidson_diagonalization(H, num_evals, tolerance=1e-8)
        
        speedup = std_time / dav_time
        speedup_str = f"{speedup:.2f}x"
        
        print(f"{size}×{size:<10} {std_time:<18.6f} {dav_time:<18.6f} {speedup_str:<15}")
    
    print(f"\n观察 (Observations):")
    print(f"  1. 小矩阵时标准方法更快 (Standard method faster for small matrices)")
    print(f"  2. 大矩阵时Davidson开始显示优势 (Davidson shows advantage for larger matrices)")
    print(f"  3. 只需少数本征值时Davidson更高效 (Davidson more efficient when few eigenvalues needed)")
    
    print("\n" + "="*80 + "\n")


def demo_accuracy_verification():
    """
    Demo 5: Verify accuracy of computed eigenpairs.
    """
    print("="*80)
    print(" "*23 + "DEMO 5: ACCURACY VERIFICATION")
    print("="*80)
    
    print("\n本示例验证计算得到的本征对的准确性")
    print("This demo verifies accuracy of computed eigenpairs.\n")
    
    size = 60
    num_evals = 3
    H = generate_hermitian_matrix(size, density=0.5, seed=123)
    
    print(f"矩阵大小 (Matrix size): {size}×{size}")
    print(f"计算本征值数量 (Number of eigenvalues): {num_evals}")
    
    # Compute eigenpairs
    dav_evals, dav_evecs, _, _ = davidson_diagonalization(H, num_evals, tolerance=1e-8)
    
    print(f"\n验证本征值方程 Hv = λv (Verifying eigenvalue equation Hv = λv):")
    print(f"\n{'本征值 Eigenvalue':<20} {'残差 ||Hv - λv||':<25} {'状态 Status':<15}")
    print("-"*80)
    
    for i in range(num_evals):
        v = dav_evecs[:, i]
        lam = dav_evals[i]
        
        # Compute Hv and λv
        Hv = H @ v
        lam_v = lam * v
        
        # Compute residual
        residual = np.linalg.norm(Hv - lam_v)
        
        status = "✓ 通过 PASS" if residual < 1e-6 else "✗ 失败 FAIL"
        print(f"λ_{i+1} = {lam:<10.6f} {residual:<25.2e} {status:<15}")
    
    # Verify orthonormality
    print(f"\n验证本征向量正交归一性 (Verifying eigenvector orthonormality):")
    gram = dav_evecs.conj().T @ dav_evecs
    identity = np.eye(num_evals)
    max_deviation = np.max(np.abs(gram - identity))
    
    print(f"  最大偏差 (Max deviation from identity): {max_deviation:.2e}")
    if max_deviation < 1e-10:
        print(f"  ✓ 本征向量是正交归一的 (Eigenvectors are orthonormal)")
    
    print("\n" + "="*80 + "\n")


def main():
    """
    Run all demos.
    """
    print("\n" + "="*80)
    print(" "*20 + "DAVIDSON DIAGONALIZATION DEMO")
    print("="*80)
    print("\n欢迎使用Davidson对角化演示程序！")
    print("Welcome to Davidson Diagonalization Demo!\n")
    print("本程序包含5个演示，展示Davidson对角化的各个方面。")
    print("This program contains 5 demos showing various aspects of Davidson diagonalization.\n")
    
    demos = [
        ("基本对比 Basic Comparison", demo_basic_comparison),
        ("矩阵性质 Matrix Properties", demo_matrix_properties),
        ("收敛分析 Convergence Analysis", demo_convergence_analysis),
        ("大小缩放 Size Scaling", demo_size_scaling),
        ("精度验证 Accuracy Verification", demo_accuracy_verification),
    ]
    
    print("可用的演示 (Available demos):")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    
    print("\n运行所有演示... (Running all demos...)\n")
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"演示失败 Demo failed: {name}")
            print(f"错误 Error: {e}\n")
    
    print("="*80)
    print(" "*30 + "DEMO COMPLETE")
    print("="*80)
    print("\n感谢使用！更多信息请参阅 DAVIDSON_README.md")
    print("Thank you! For more information, see DAVIDSON_README.md\n")


if __name__ == "__main__":
    main()

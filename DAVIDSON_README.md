# Davidson Diagonalization Comparison

## 概述 (Overview)

本项目实现了厄米矩阵对角化的两种方法的比较：
1. **标准对角化方法** (Standard Diagonalization) - 使用NumPy/SciPy的全对角化
2. **Davidson对角化方法** (Davidson Diagonalization) - 迭代方法，适用于大型稀疏矩阵

This project implements and compares two methods for diagonalizing Hermitian matrices:
1. **Standard Diagonalization** - Full diagonalization using NumPy/SciPy
2. **Davidson Diagonalization** - Iterative method suitable for large sparse matrices

## 文件说明 (Files)

### 核心实现 (Core Implementation)
- **`davidson_comparison.py`** - 主要比较脚本，包含完整的实现
  - 生成随机厄米矩阵
  - 实现Davidson对角化算法
  - 对比两种方法的结果
  - Main comparison script with complete implementation

### 示例与测试 (Examples and Benchmarks)
- **`example_davidson.py`** - 使用示例和教程
  - 基本使用示例
  - 自定义参数示例
  - 收敛性研究
  - Usage examples and tutorials

- **`davidson_benchmark.py`** - 性能基准测试
  - 多个矩阵大小的对比测试
  - 性能统计分析
  - Performance benchmarking across different matrix sizes

## 快速开始 (Quick Start)

### 安装依赖 (Install Dependencies)

```bash
pip install numpy scipy
```

### 基本使用 (Basic Usage)

运行基本比较：
```bash
python3 davidson_comparison.py
```

运行示例：
```bash
python3 example_davidson.py
```

运行基准测试：
```bash
python3 davidson_benchmark.py
```

### 自定义参数 (Custom Parameters)

使用自定义矩阵大小：
```bash
python3 example_davidson.py 150 5
# 150: 矩阵大小 (matrix size)
# 5: 计算的本征值数量 (number of eigenvalues)
```

基准测试多个大小：
```bash
python3 davidson_benchmark.py 50,100,200 3
# 50,100,200: 要测试的矩阵大小 (matrix sizes to test)
# 3: 每个大小的试验次数 (number of trials per size)
```

## 算法说明 (Algorithm Details)

### Davidson对角化算法 (Davidson Diagonalization Algorithm)

Davidson方法是一种迭代算法，特别适用于大型稀疏厄米矩阵的部分对角化。

The Davidson method is an iterative algorithm particularly efficient for partial diagonalization of large sparse Hermitian matrices.

**主要步骤 (Main Steps):**

1. **初始化** (Initialization)
   - 生成随机初始向量
   - 正交归一化

2. **迭代过程** (Iteration Process)
   - 投影矩阵到子空间
   - 对角化投影矩阵
   - 计算Ritz向量和残差
   - 检查收敛性
   - 添加修正向量扩展子空间

3. **收敛判据** (Convergence Criteria)
   - 残差范数 < 容差 (tolerance)
   - 或达到最大迭代次数

**优势 (Advantages):**
- 只计算需要的几个本征值，而不是全部
- 对大型稀疏矩阵非常高效
- 内存需求较低
- Only computes needed eigenvalues, not all
- Very efficient for large sparse matrices
- Lower memory requirements

**劣势 (Disadvantages):**
- 对小矩阵可能比标准方法慢
- 需要调整参数（容差、最大迭代次数）
- May be slower than standard methods for small matrices
- Requires parameter tuning (tolerance, max iterations)

## 实现细节 (Implementation Details)

### 厄米矩阵生成 (Hermitian Matrix Generation)

```python
def generate_hermitian_matrix(size, density=0.5, seed=None):
    """
    生成随机厄米矩阵
    Generate a random Hermitian matrix
    
    参数 Parameters:
    - size: 矩阵大小 (matrix size)
    - density: 非零元素密度 (density of non-zero elements)
    - seed: 随机种子 (random seed for reproducibility)
    """
```

生成方法：
1. 创建随机复数矩阵 A
2. 使用公式 H = (A + A†) / 2 确保厄米性
3. 应用稀疏性掩码

Generation method:
1. Create random complex matrix A
2. Ensure Hermitian property using H = (A + A†) / 2
3. Apply sparsity mask

### 对比指标 (Comparison Metrics)

1. **本征值差异** (Eigenvalue Difference)
   ```
   Δλ = |λ_standard - λ_davidson|
   ```

2. **本征向量重叠** (Eigenvector Overlap)
   ```
   overlap = |⟨v_standard | v_davidson⟩|
   ```
   完美匹配时应接近 1.0 (Should be close to 1.0 for perfect match)

3. **计算时间** (Computation Time)
   - 标准方法时间 (Standard method time)
   - Davidson方法时间 (Davidson method time)
   - 迭代次数 (Number of iterations)

## 结果示例 (Example Results)

### 100×100 矩阵 (100×100 Matrix)

```
Matrix Size: 100 × 100

Method               Time (s)        Iterations     
----------------------------------------------------------------------
Standard             0.009698        N/A            
Davidson             0.053917        59             

Eigenstate      Standard             Davidson             Difference          
----------------------------------------------------------------------
1               -13.3871751675       -13.3871751675       3.55e-15            
2               -13.0198388253       -13.0198388253       0.00e+00            
3               -12.1585173403       -12.1585173403       1.78e-15            

Eigenvalue Comparison:
  - Maximum difference: 3.55e-15
  - Mean difference:    1.78e-15

Eigenvector Overlap:
  - Minimum overlap:  1.0000000000
  - Mean overlap:     1.0000000000

CONCLUSION:
✓ Excellent agreement between standard and Davidson methods!
```

## 性能分析 (Performance Analysis)

### 矩阵大小 vs 效率 (Matrix Size vs Efficiency)

| 矩阵大小<br>Size | 标准方法<br>Standard (s) | Davidson方法<br>Davidson (s) | 加速比<br>Speedup | 迭代次数<br>Iterations |
|---------|------------|------------|----------|-----------|
| 50×50   | 0.0013     | 0.0240     | 0.06x    | 40        |
| 100×100 | 0.0024     | 0.0235     | 0.10x    | 61        |
| 200×200 | ~0.020     | ~0.040     | ~0.50x   | ~90       |
| 500×500 | ~0.300     | ~0.200     | ~1.50x   | ~150      |

**观察 (Observations):**
- 小矩阵：标准方法更快
- 大矩阵：Davidson方法开始显示优势
- 只需要少数本征值时，Davidson方法效率更高

- Small matrices: Standard method is faster
- Large matrices: Davidson method shows advantages
- When only few eigenvalues needed, Davidson is more efficient

## 应用场景 (Application Scenarios)

### 适合使用Davidson方法 (Good for Davidson Method)
- 大型稀疏矩阵 (Large sparse matrices)
- 只需要少数本征值 (Only need few eigenvalues)
- 内存受限环境 (Memory-constrained environments)
- 量子化学计算 (Quantum chemistry calculations)
- 固体物理问题 (Solid state physics problems)

### 适合使用标准方法 (Good for Standard Method)
- 小型矩阵 (Small matrices, < 500×500)
- 需要所有本征值 (Need all eigenvalues)
- 稠密矩阵 (Dense matrices)
- 快速原型开发 (Rapid prototyping)

## 参数调优建议 (Parameter Tuning Recommendations)

### Davidson方法参数 (Davidson Method Parameters)

```python
davidson_diagonalization(
    H,                          # 厄米矩阵 (Hermitian matrix)
    num_eigenvalues=3,         # 本征值数量 (number of eigenvalues)
    max_iterations=1000,        # 最大迭代次数 (max iterations)
    tolerance=1e-8              # 收敛容差 (convergence tolerance)
)
```

**容差选择 (Tolerance Selection):**
- `1e-4`: 快速但精度较低 (Fast but lower accuracy)
- `1e-6`: 平衡选择 (Balanced choice)
- `1e-8`: 高精度（推荐）(High accuracy - recommended)
- `1e-10`: 极高精度，可能需要更多迭代 (Very high accuracy, may need more iterations)

**最大迭代次数 (Max Iterations):**
- 小矩阵 (<100): 100-500
- 中等矩阵 (100-500): 500-1000
- 大矩阵 (>500): 1000-2000

## 技术细节 (Technical Details)

### 预处理器 (Preconditioner)

当前实现使用简单的对角预处理器：
```python
diag_H = np.diag(H)
corrections[:, i] = -residuals[:, i] / (diag_H - theta[i])
```

可以改进为更复杂的预处理器以提高收敛速度。

Current implementation uses a simple diagonal preconditioner.
Can be improved with more sophisticated preconditioners for faster convergence.

### 子空间扩展 (Subspace Expansion)

为防止子空间过大，实现了重启机制：
```python
if V.shape[1] > num_eigenvalues * 10:
    V = ritz_vectors[:, :num_eigenvalues * 2]
```

To prevent subspace from growing too large, restart mechanism is implemented.

## 扩展与改进 (Extensions and Improvements)

### 可能的改进方向 (Possible Improvements)

1. **并行化** (Parallelization)
   - 使用多线程加速矩阵乘法
   - Use multi-threading for matrix operations

2. **更好的预处理器** (Better Preconditioners)
   - 不完全LU分解 (Incomplete LU factorization)
   - 多重网格方法 (Multigrid methods)

3. **自适应参数** (Adaptive Parameters)
   - 根据矩阵性质自动调整容差
   - Automatically adjust tolerance based on matrix properties

4. **GPU加速** (GPU Acceleration)
   - 使用CuPy或JAX实现GPU版本
   - Implement GPU version using CuPy or JAX

## 参考文献 (References)

1. Davidson, E. R. (1975). "The iterative calculation of a few of the lowest eigenvalues and corresponding eigenvectors of large real-symmetric matrices". Journal of Computational Physics.

2. Sleijpen, G. L., & Van der Vorst, H. A. (1996). "A Jacobi–Davidson iteration method for linear eigenvalue problems". SIAM Journal on Matrix Analysis and Applications.

## 许可证 (License)

本代码是OPTXC项目的一部分，遵循项目的许可证。

This code is part of the OPTXC project and follows the project's license.

## 联系方式 (Contact)

如有问题或建议，请通过GitHub Issues联系。

For questions or suggestions, please contact through GitHub Issues.

---

**注意**: 本实现主要用于教学和演示目的。对于生产环境，建议使用成熟的库如SciPy、ARPACK或FEAST。

**Note**: This implementation is primarily for educational and demonstration purposes. For production use, consider mature libraries like SciPy, ARPACK, or FEAST.

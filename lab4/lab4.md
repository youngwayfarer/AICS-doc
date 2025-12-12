# Lab4

## 实验目的

- 了解 TVM 深度学习编译器
- 掌握利用 Tensor Expression(TE) 定义算子、构建 IRModule 并生成可执行库的流程
- 学习以矩阵乘法（GEMM）为例逐步应用调度优化的方法

## 环境配置

本次实验需要安装 TVM。有以下几种方式：

### 2.1 Pip 安装 TVM

通过 pip 安装 TVM，适合快速体验 TVM，但可能无法使用最新特性，且无法自定义编译选项。如果只在本次实验中使用 TVM，推荐此方法。

1. 手动访问 https://mlc.ai/wheels (需要代理)，选择适合你系统的预编译包下载。例如，对于 Ubuntu 系统，只在 CPU 上使用 TVM，可以下载mlc_ai_nightly_cpu-0.20.dev650-py3-none-manylinux_2_28_x86_64.whl。

2. 使用 pip 安装下载的 whl 文件，例如：

   ```bash
   pip install mlc_ai_nightly_cpu-0.20.dev650-py3-none-manylinux_2_28_x86_64.whl
   ```

### 2.2 从源码编译 TVM

从源码编译 TVM 可以自定义编译选项，并且可以手动修改源码以实现特定功能。

访问 https://tvm.apache.org/docs/install/from_source.html ，根据指导完成编译。

### 2.3 Docker 安装 TVM

使用 Docker 可以快速搭建 TVM 环境。

访问 https://tvm.apache.org/docs/install/docker.html ，根据指导完成安装。

> 注意：由于 TVM 版本更新较快，并且不同版本 API 可能存在差异。本次实验推荐使用第一种方式安装 TVM，版本为 0.20，如果选择其他版本可能会遇到兼容性问题，需要修改部分代码以适配。

## TVM 相关基础知识

TVM 是一个开源的深度学习编译器栈，旨在将深度学习模型高效地部署到各种硬件平台上。TVM 主要包含以下几个核心组件：
- **前端（Front-end）**：支持多种深度学习框架（如 TensorFlow、PyTorch、MXNet 等）的模型导入，将模型转换为中间表示（IR）。
- **中间表示（IR）**：TVM 使用多层次的中间表示来描述计算图和调度信息。
    - **图层级 IR**：用于表示高层次的计算图，支持函数式编程风格。
      - Relay
      - Relax
    - **算子层 IR（TE）**：用于描述具体的算子计算和调度。
      - Tensor Expression
      - Tensor IR

- **调度器（Scheduler）**：提供丰富的调度原语，允许用户对计算图进行优化，如循环变换、内存布局优化等。
- **后端（Back-end）**：将优化后的中间表示编译为目标硬件平台的可执行代码，支持 CPU、GPU 以及专用加速器等多种硬件。

本次实验主要介绍算子层 IR，Tensor Expression（TE）的相关内容，以及如何对算子进行优化。

对于其他部分感兴趣以及需要查询不同方法用法的同学可以参考 TVM 官方文档：https://tvm.apache.org/docs/ ，以及网络相关课程内容：https://mlc.ai/zh/chapter_introduction/index.html 。

## TVM 中的 GEMM 基础实现

矩阵乘法 `(M, K) x (K, N)` 是衡量深度学习编译器性能的经典算子。`gemm.ipynb` 先利用 NumPy 计算基线结果，再用 TVM TE 重新实现 GEMM 并验证正确性。

为了简化，本次实现的 GEMM 算子不包含偏置，仅实现最基本的矩阵乘法功能。

### 3.1 利用 TE 定义计算

```python
M = N = K = 1024
dtype = "float32"

# 规约轴
k = te.reduce_axis((0, K), "k")

# 定义两个输入矩阵 A、B
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")

# 定义输出矩阵 C 以及计算表达式
C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

# 定义原语函数并创建 IRModule
prim_func = te.create_prim_func([A, B, C]).with_attr("global_symbol", "mmult")
mod = tvm.IRModule({"mmult": prim_func})
```

`mod` 是一个包含 `mmult` 函数的 IRModule，可以通过`print`来查看生成的 Tensor IR 代码。

如此时`mod`打印结果如下：

``` 
@I.ir_module
class Module:
    @T.prim_func
    def mmult(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for x, y, k in T.grid(1024, 1024, 1024):
            with T.block("C"):
                v_x, v_y, v_k = T.axis.remap("SSR", [x, y, k])
                T.reads(A[v_x, v_k], B[v_k, v_y])
                T.writes(C[v_x, v_y])
                with T.init():
                    C[v_x, v_y] = T.float32(0.0)
                C[v_x, v_y] = C[v_x, v_y] + A[v_x, v_k] * B[v_k, v_y]
```

Tensor IR 代码以 block 为单位组织，`T.grid` 定义三重循环，`T.axis.remap` 定义轴的类型（S：空间轴，R：规约轴），`T.reads` 和 `T.writes` 分别声明读写的缓冲区。

可以看到对应的代码结构与我们熟悉的三重循环矩阵乘法类似，通过三重循环遍历输出矩阵 `C` 的每个元素，并对 `k` 轴进行规约累加，来实现矩阵乘法。

随后通过 `tvm.build` 生成目标平台（示例中使用 `llvm`）的目标代码：

```python
# 定义目标平台，此处为 CPU
target = tvm.target.Target(target="llvm", host="llvm")
dev = tvm.device(target.kind.name, 0)

# 生成目标平台上的可执行代码
lib = tvm.build(mod, target=target)
# func 即为可调用的矩阵乘法函数
func = lib["mmult"]

c = tvm.runtime.tensor(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
```

这里需要注意的是，TVM 中的张量可以使用 `tvm.runtime.tensor` 创建，并传入 NumPy 张量和设备信息。
如`c = tvm.runtime.tensor(numpy.zeros((M, N), dtype=dtype), dev)`。同时还可以转换回 NumPy 张量：`c.numpy()`。

### 3.2 评测函数

Notebook 中封装了 `evaluate_operation`，用于运行 `time_evaluator` 并记录每个调度的平均耗时：

```python
def evaluate_operation(lib, optimization, log):
    func = lib["mmult"]
    c = tvm.runtime.tensor(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)
    evaluator = lib.time_evaluator(lib.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print(f"{optimization}: {mean_time}")
    log.append((optimization, mean_time))
```

此函数会在后续六种优化后重复调用，以便对比性能。

## GEMM 的优化策略

在基础计算图生成后，可以通过 `tvm.tir.Schedule` 对应 `block` 和 `loop` 的操作来逐步提升性能。以下所有示例都可以在 `gemm.ipynb` 中找到完整代码。

本节介绍六种常见的手动优化策略。主要是针对计算和内存访问模式进行优化，以提升缓存命中率、利用向量化指令和多核并行能力。

当然，TVM 也支持自动调度来自动搜索最优调度策略，但本次实验中为了大家更理解底层优化原理，我们主要介绍手动调度方法。感兴趣的同学可以参考 TVM 官方文档中关于自动调度的部分内容。

> 需要注意的是，以下几种优化方法涉及循环重排等，这并不一定适用于所有循环嵌套结构。为什么？

### 4.1 Blocking

当前算法对 `M`、`N` 轴顺序扫描，在原始的三重循环中，计算大矩阵（如 1024x1024）时，当我们在处理矩阵的一行时，由于数据量超过了 L1 缓存的大小（通常仅 32KB），等到下一次需要复用某些数据时，它们已经被“挤出”了缓存。
因此 CPU 不得不频繁地从速度较慢的 RAM 中重新读取数据，导致计算单元大部分时间在等待数据，而非进行计算。

为了解决这个问题，可以将输出划分为 32×32 的小块，这一小块需要的 A、B 的对应数据可以在缓存中停留整个 tile 的计算过程，从而显著提升局部性。

具体做法：

- `xo/xi`、`yo/yi` 拆分外层循环后，每次只处理一个 tile，可保证 tile 数据可以完整放入缓存；
- `ko/ki` 将规约轴拆分为更小的累加深度，进一步增强 `A[x, k]`、`B[k, y]` 的重用；
- `reorder(xo, yo, ko, ki, xi, yi)` 让外层先确定 tile，再在 tile 内做规约，确保被 tile 选中的数据在缓存中的停留时间最长。

```python
bn = 32

# 创建调度对象
sch = tvm.tir.Schedule(mod)

# 获取 C block 及其循环
block_c = sch.get_block("C", func_name="mmult")
x, y, k = sch.get_loops(block_c)

# 进行 blocking
xo, xi = sch.split(x, factors=[None, bn])
yo, yi = sch.split(y, factors=[None, bn])
ko, ki = sch.split(k, factors=[None, 4])

# 重新排序循环顺序
sch.reorder(xo, yo, ko, ki, xi, yi)
```

完成 blocking 后，每次只处理一个 `32x32` 的 tile 并在该 tile 上完成 K 方向的累加，相比初始算法能显著改善缓存命中率，为后续的矢量化和并行化奠定基础。

此时查看对应的 Tensor IR 代码，可以看到循环结构已经发生了变化：

```
@I.ir_module
class Module:
    @T.prim_func
    def mmult(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for x_0, y_0, k_0, k_1, x_1, y_1 in T.grid(32, 32, 256, 4, 32, 32):
            with T.block("C"):
                v_x = T.axis.spatial(1024, x_0 * 32 + x_1)
                v_y = T.axis.spatial(1024, y_0 * 32 + y_1)
                v_k = T.axis.reduce(1024, k_0 * 4 + k_1)
                T.reads(A[v_x, v_k], B[v_k, v_y])
                T.writes(C[v_x, v_y])
                with T.init():
                    C[v_x, v_y] = T.float32(0.0)
                C[v_x, v_y] = C[v_x, v_y] + A[v_x, v_k] * B[v_k, v_y]
```

可以发现，外层循环已经变为 `x_0, y_0, k_0, k_1, x_1, y_1`，每次处理一个 `32x32` 的 tile，并在该 tile 上完成 K 方向的累加。

### 4.2 Vectorization

当前代码仍然是标量（Scalar）操作，即 CPU 一个时钟周期只能处理一个浮点数加法或乘法。现代 CPU（如支持 AVX2/AVX-512 的 CPU）具备宽向量寄存器，可以进行向量操作，不使用这些特性相当于浪费了大部分算力。

在上一节的调度基础上，只需要对 `yi` 调用 `vectorize`：

```python
# 向量化
sch.vectorize(yi)
```

这会将最内层循环映射为向量化指令，能显著减少指令数量，提高计算吞吐量。

查看对应的 Tensor IR 代码，可以看到最内层循环已经被替换为向量化指令：

```
@I.ir_module
class Module:
    @T.prim_func
    def mmult(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for x_0, y_0, k_0, k_1, x_1 in T.grid(32, 32, 256, 4, 32):
            for y_1 in T.vectorized(32):
                with T.block("C"):
                    v_x = T.axis.spatial(1024, x_0 * 32 + x_1)
                    v_y = T.axis.spatial(1024, y_0 * 32 + y_1)
                    v_k = T.axis.reduce(1024, k_0 * 4 + k_1)
                    T.reads(A[v_x, v_k], B[v_k, v_y])
                    T.writes(C[v_x, v_y])
                    with T.init():
                        C[v_x, v_y] = T.float32(0.0)
                    C[v_x, v_y] = C[v_x, v_y] + A[v_x, v_k] * B[v_k, v_y]
```

> 思考：在上一小节优化的基础上，如果只对最内层进行向量化，得到的性能一定会有所提升吗？为什么？如果没有提升，可能的原因有哪些？

### 4.3 Loop Permutation

对矩阵 B 的访问做到了顺序访问，接下来考虑矩阵 A 的访问，当前对于 A 的访问是按列访问的，而 CPU 的缓存行通常是按行存储的，这会导致大量的缓存 miss。无法利用缓存行的空间局部性，影响性能。

因此可以尝试调整循环顺序，使得对 A 的访问也变为顺序访问。具体思路比较简单，将迭代变量`xi`和`ki`交换位置，从而使得对 A 的访问变为按行访问即可。

这里直接从初始算法开始：

```python
# 创建调度对象，从初始算法开始
sch = tvm.tir.Schedule(mod)

# 获取 C block 及其循环
block_c = sch.get_block("C", func_name="mmult")
x, y, k = sch.get_loops(block_c)

# 进行 blocking
xo, xi = sch.split(x, factors=[None, bn])
yo, yi = sch.split(y, factors=[None, bn])
ko, ki = sch.split(k, factors = [None, 4])

# 重新排序循环顺序
sch.reorder(xo, yo, ko, xi, ki, yi)

# 向量化
sch.vectorize(yi)
```

相比 Blocking 阶段的 `xo, yo, ko, ki, xi, yi`，这里将 `xi` 向前移动，以便同一行块内的计算连续执行，提升对 A 的访问局部性。

查看对应的 Tensor IR：

```
@I.ir_module
class Module:
    @T.prim_func
    def mmult(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for x_0, y_0, k_0, x_1, k_1 in T.grid(32, 32, 256, 32, 4):
            for y_1 in T.vectorized(32):
                with T.block("C"):
                    v_x = T.axis.spatial(1024, x_0 * 32 + x_1)
                    v_y = T.axis.spatial(1024, y_0 * 32 + y_1)
                    v_k = T.axis.reduce(1024, k_0 * 4 + k_1)
                    T.reads(A[v_x, v_k], B[v_k, v_y])
                    T.writes(C[v_x, v_y])
                    with T.init():
                        C[v_x, v_y] = T.float32(0.0)
                    C[v_x, v_y] = C[v_x, v_y] + A[v_x, v_k] * B[v_k, v_y]
```

### 4.4 Array Packing

注意到上一步优化后的 Tensor IR 中，对于 B 的访问看起来是连续的，考虑最内层的两层循环`k_1` 和 `y_1`，固定外层四个循环变量，我们来查看对于 B 的访问模式：首先在行上连续访问 32 个元素，然后一旦`k_1`增加 1，即 `v_k` 增加 1，在矩阵 B 中就会跳转到下一行，也就是说对于 B 的访问模式实际上是“行跳跃”的，为了在 32×32 的小方块里往下挪一行，内存指针必须跳过 1024 个位置，这样会导致大量缓存 miss，影响性能。

为了解决这个问题，可以按照访问模式，直接将矩阵 B 给打包成对应的形状，即将 B 从 1024×1024 打包成 32×1024×32 的形状，这样在计算时就能真正做到对 B 的顺序访问，提升缓存命中率。

实现代码如下：

```python
# 首先使用 te 矩阵 B 进行打包
packedB = te.compute((N // bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")

# 接着利用 packedB 来计算 C
C: Any = te.compute(
    (M, N),
    lambda x, y: te.sum(A[x, k] * packedB[tvm.tir.floordiv(y, bn), k, tvm.tir.indexmod(y, bn)], axis=k),
    name="C",
)

# 创建原语函数和 IRModule
prim_func = te.create_prim_func([A, B, C]).with_attr("global_symbol", "mmult")
mod = tvm.IRModule({"mmult": prim_func})

# 创建调度对象
sch = tvm.tir.Schedule(mod)

# 获取 C block 及其循环
block_c = sch.get_block("C", func_name="mmult")
x, y, k = sch.get_loops(block_c)

# 进行 blocking
xo, xi = sch.split(x, factors=[None, bn])
yo, yi = sch.split(y, factors=[None, bn])
ko, ki = sch.split(k, factors=[None, 4])

# 重新排序循环顺序以及向量化
sch.reorder(xo, yo, ko, xi, ki, yi)
sch.vectorize(yi)
```

需要注意的是，以上代码中关于循环拆分重排等优化均是针对块 C 而言，对于矩阵 B 的打包操作并没有进行优化，而打包操作本身也是循环，并且没有相关数据依赖关系，可以进行并行化向量化等优化：

```python
# 获取 packedB 相关 block 及其循环
block_pack = sch.get_block("packedB", func_name="mmult")
xp, yp, zp = sch.get_loops(block_pack)

# 对 packedB 进行向量化和并行化
sch.vectorize(zp)
sch.parallel(xp)
```

这样每个线程负责打包若干列块，并且打包结果可被所有后续 tile 重用。

当前的 Tensor IR 如下：

```
@I.ir_module
class Module:
    @T.prim_func
    def mmult(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        packedB = T.alloc_buffer((32, 1024, 32))
        for x in T.parallel(32):
            for y in range(1024):
                for z in T.vectorized(32):
                    with T.block("packedB"):
                        v_x, v_y, v_z = T.axis.remap("SSS", [x, y, z])
                        T.reads(B[v_y, v_x * 32 + v_z])
                        T.writes(packedB[v_x, v_y, v_z])
                        packedB[v_x, v_y, v_z] = B[v_y, v_x * 32 + v_z]
        for x_0, y_0, k_0, x_1, k_1 in T.grid(32, 32, 256, 32, 4):
            for y_1 in T.vectorized(32):
                with T.block("C"):
                    v_x = T.axis.spatial(1024, x_0 * 32 + x_1)
                    v_y = T.axis.spatial(1024, y_0 * 32 + y_1)
                    v_k = T.axis.reduce(1024, k_0 * 4 + k_1)
                    T.reads(A[v_x, v_k], packedB[v_y // 32, v_k, v_y % 32])
                    T.writes(C[v_x, v_y])
                    with T.init():
                        C[v_x, v_y] = T.float32(0.0)
                    C[v_x, v_y] = C[v_x, v_y] + A[v_x, v_k] * packedB[v_y // 32, v_k, v_y % 32]
```

可以看到当前代码中主要有两个 block，分别是 `packedB` 和 `C`。`packedB` block 中的循环已经被并行化和向量化，而 `C` block 则保持之前的优化。但通过打包操作，矩阵 B 的访问模式已经变为顺序访问，提升了缓存命中率。

### 4.5 写回缓存（Cache Write）

前面几个优化主要针对的是计算和读取操作，即如何提高缓存的利用率，更高效地读取矩阵 A、B 并进行计算。接下来考虑写回操作。

当前代码中，每次对于目标 `C[v_x, v_y]` 的更新，都需要从全局内存中读取该位置的值，进行累加后再写回，这样会导致大量的内存读写操作，影响性能。

为了减少对于内存的读写次数，可以考虑将输出块先保存在高速缓存中，等到整个计算过程完成后再一次性写回全局内存。这样可以显著减少内存带宽的压力，提高整体性能。

而实际执行中，由于缓存大小的限制，可以考虑完成一个 tile 的计算后写回一次，而不是等到整个矩阵计算完成后再写回。

实现代码如下：

```python
# 创建调度对象，从初始算法开始
sch = tvm.tir.Schedule(mod)

# 获取 C block 及其循环
block_c = sch.get_block("C", func_name="mmult")
x, y, k = sch.get_loops(block_c)

# 创建写回缓存
CC = sch.cache_write(block_c, 0, "global")

# 进行 blocking
xo, xi = sch.split(x, factors=[None, bn])
yo, yi = sch.split(y, factors=[None, bn])
ko, ki = sch.split(k, factors=[None, 4])

# 重新排序循环顺序
sch.reorder(xo, yo, ko, xi, ki, yi)

# 对于写回缓存 CC 进行循环展开和向量化
xc, yc = sch.get_loops(CC)[-2:]
sch.unroll(ki)
sch.vectorize(yc)

# 将写回操作放到 yo 循环下
sch.reverse_compute_at(CC, yo)
```

`reverse_compute_at` 将写回操作放到 `yo` 循环下执行，保证完成一个 tile 的计算后再写回，从而减少内存读写次数，并且不会出现缓存溢出的问题。

查看对应的 Tensor IR：

```
@I.ir_module
class Module:
    @T.prim_func
    def mmult(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        packedB = T.alloc_buffer((32, 1024, 32))
        C_global = T.alloc_buffer((1024, 1024))
        for x, y, z in T.grid(32, 1024, 32):
            with T.block("packedB"):
                v_x, v_y, v_z = T.axis.remap("SSS", [x, y, z])
                T.reads(B[v_y, v_x * 32 + v_z])
                T.writes(packedB[v_x, v_y, v_z])
                packedB[v_x, v_y, v_z] = B[v_y, v_x * 32 + v_z]
        for x_0, y_0 in T.grid(32, 32):
            for k_0, x_1 in T.grid(256, 32):
                for k_1 in T.unroll(4):
                    for y_1 in range(32):
                        with T.block("C"):
                            v_x = T.axis.spatial(1024, x_0 * 32 + x_1)
                            v_y = T.axis.spatial(1024, y_0 * 32 + y_1)
                            v_k = T.axis.reduce(1024, k_0 * 4 + k_1)
...
                    v1 = T.axis.spatial(1024, y_0 * 32 + ax1)
                    T.reads(C_global[v0, v1])
                    T.writes(C[v0, v1])
                    C[v0, v1] = C_global[v0, v1]
```

可以看到此时已经加入了写缓存，每完成一个 tile 的计算后，才将结果写回全局内存 `C`。

### 4.6 Parallelization

目前已经对计算和内存访问进行了多种优化，接下来考虑利用多核 CPU 的并行计算能力，进一步提升性能。（实际上并行化在前面已经使用过了）

```python
# 获取 C block 及其循环
block_c = sch.get_block("C", "mmult")
xo = sch.get_loops(block_c)[0]
# 对 xo 进行并行化
sch.parallel(xo)

# 获取 packedB 相关 block 及其循环
block_pack = sch.get_block("packedB", "mmult")
xp, yp, zp = sch.get_loops(block_pack)
# 对 packedB 进行并行化以及向量化
sch.parallel(xp)
sch.vectorize(zp)
```

这里将 `xo`（行块）作为并行维度，使不同核心处理不同的 M 方向 tile；对 `packedB` 的 `xp` 并行，则能够更快完成预处理。

此处由于 Tensor IR 代码变化不大，不再给出，可以运行`gemm.ipynb` 中的相关代码查看结果。

## 性能汇总

接下来给出几种优化带来的性能提升：

| Operator          | Timing (s)        | Performance (Relative) |
|-------------------|-------------------|------------------------|
| none              | 5.6712718502      | 1.0                    |
| blocking          | 0.2820879918      | 0.049739811324694655   |
| vectorization     | 0.3215404293      | 0.056696352739405494   |
| loop permutation  | 0.0992255179      | 0.01749616673665551    |
| array packing     | 0.1404239226      | 0.02476056981734139    |
| write cache       | 0.4168496218      | 0.07350196442889607    |
| write cache       | 0.0282441346      | 0.004980211731342761   |

可以发现经过多种联合优化后，性能提升显著！从最初的 5.67 秒降至 0.028 秒，提升了约 200 倍。

## 实验内容

`attention.ipynb`中给出了点积注意力机制的基础实现，请基于以上介绍的几种优化方法（完整代码见`gemm.ipynb`），实现对 attention 算子的优化。

要求：
- 逐步优化
- 每次优化后调用 `evaluate_operation` 评测性能
- 最终打印出`log`中的所有结果

你需要在实验报告中解释每个优化步骤的设计思路，并分析每次优化带来的性能变化。

## 提交要求

将包含所有单元格运行结果的`attention.ipynb`文件与实验报告打包在一起并压缩到 zip 格式，命名格式为 `学号_姓名_lab4.zip`，上传到 bb 平台。

<span style="color:red; font-weight:bold;">提交截止时间</span>：北京时间 1 月 5 日 23:59。

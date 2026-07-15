# BP3333 Airfoil Parameterization

This repository provides a Python implementation of the BP3333 method presented in **Bezier-PARSEC: An optimized aerofoil parameterization for design**.

BP3333 uses four cubic Bézier curves to describe the leading and trailing portions of the thickness distribution and camber line. The airfoil is controlled by 12 parameters with aerodynamic and geometric meaning. This implementation supports fitting BP3333 parameters to discrete airfoil coordinates, reconstructing the airfoil, calculating fitting errors, and generating comparison plots.

## Requirements

- Python 3.10+
- NumPy
- SciPy
- Matplotlib

Install the dependencies with:

```bash
python -m pip install numpy scipy matplotlib
```

## Quick Start

First enter the `BP3333` directory. All commands below run directly from this
directory:

```bash
cd /path/to/BP3333
```

Fit a single airfoil:

```bash
python main.py \
  --airfoil "Test Airfoils/n0012.dat" \
  --output-dir "results"
```

Fit all airfoils in the test directory:

```bash
python main.py \
  --all \
  --test-dir "Test Airfoils" \
  --output-dir "results"
```

Multi-start SLSQP is used by default. If every SLSQP run fails, the code
automatically falls back to a Differential Evolution GA. GA and nonlinear
least-squares optimization can also be selected directly:

```bash
python main.py \
  --airfoil "Test Airfoils/n0012.dat" \
  --output-dir "results" \
  --optimizer ga
```

### Surface Pairing and Chord Normalization

Three upper/lower surface matching modes are available:

- `--pairing x`: interpolate both surfaces onto the same cosine-spaced `x` array, regardless of their original point counts;
- `--pairing arc`: independently resample both surfaces by normalized arc length, regardless of their original point counts;
- `--pairing auto` (default): use common `x` when both surfaces are monotone single-valued `y(x)` curves, and use normalized arc length when a rounded leading or trailing edge creates an `x` overhang.

Chord normalization can likewise be selected explicitly or automatically:

```bash
python main.py \
  --airfoil "Test Airfoils/n0012.dat" \
  --pairing auto \
  --normalise-chord auto
```

`--normalise-chord yes` forces translation, rotation, and scaling onto the
`[0,0]`--`[1,0]` chord. `--normalise-chord no` preserves the input coordinates.
The default `auto` mode checks the inferred chord length, direction, and LE/TE
x locations and avoids transforming coordinates already expressed on a unit chord.


LE and virtual TE coordinates can be set explicitly:

```bash
python main.py \
  --airfoil "path/to/truncated_blunt_te.dat" \
  --normalise-chord yes \
  --leading-edge 0.0 0.0 \
  --trailing-edge 1.0 0.0
```


## Input and Output

The input file must contain airfoil contour coordinates. The first line is the airfoil name, followed by one `x y` coordinate pair per line. The contour should be ordered from the trailing edge to the leading edge along the upper surface, then from the leading edge to the trailing edge along the lower surface.

Each fit produces:

- `*_bp3333.dat`: reconstructed airfoil coordinates;
- `*_bp3333.json`: the 12 BP3333 parameters, errors, optimizer information, and root-selection strategy;
- `*_bp3333_thickness.png`: comparison of the thickness and camber distributions;
- `*_bp3333_airfoil.png`: comparison of the reference and fitted airfoil contours.

## Project Structure

- `model.py`: BP3333 parameters, Bézier control-point calculations, and airfoil generation;
- `fitting.py`: parameter initialization, constraints, and numerical fitting;
- `geometry.py`: airfoil coordinate input and geometric feature extraction;
- `main.py`: command-line interface;
- `io.py` / `plotting.py`: result export and plotting utilities.

## Reference

R. W. Derksen and T. Rogalsky, “Bezier-PARSEC: An optimized aerofoil parameterization for design,” *Advances in Engineering Software*, vol. 41, pp. 923-930, 2010. DOI: [10.1016/j.advengsoft.2010.05.002](https://doi.org/10.1016/j.advengsoft.2010.05.002)


# BP3333 翼型参数化

这是论文 **Bezier-PARSEC: An optimized aerofoil parameterization for design** 中 BP3333 方法的 Python 实现。

BP3333 使用 4 段三次 Bézier 曲线分别描述翼型的前缘/后缘厚度分布和前缘/后缘弯度线，并通过 12 个具有气动几何意义的参数控制翼型。代码支持从离散翼型坐标拟合 BP3333 参数、重建翼型，以及输出拟合误差和对比图。

## 环境

- Python 3.10+
- NumPy
- SciPy
- Matplotlib

安装依赖：

```bash
python -m pip install numpy scipy matplotlib
```

## 快速开始

拟合单个翼型：

```bash
python main.py \
  --airfoil "Test Airfoils/n0012.dat" \
  --output-dir "results"
```

拟合测试目录中的全部翼型：

```bash
python main.py \
  --all \
  --test-dir "Test Airfoils" \
  --output-dir "results"
```

默认使用多起点 SLSQP；如果所有 SLSQP 计算均失败，程序会自动改用
GA。也可以直接选择 GA 或非线性最小二乘：

```bash
python main.py \
  --airfoil "Test Airfoils/n0012.dat" \
  --output-dir "results" \
  --optimizer ga
```


### 表面配对与弦长归一化

上下表面配对有三种模式：

- `--pairing x`：分别插值到同一组余弦分布的 `x` 坐标；
- `--pairing arc`：分别按各自的归一化弧长重采样后配对；
- `--pairing auto`（默认）：两侧均为单调、单值的 `y(x)` 时选择 `x`，检测到圆钝前/尾缘回卷时选择 `arc`。

弦长坐标也支持显式或自动控制：

```bash
python main.py \
  --airfoil "Test Airfoils/n0012.dat" \
  --pairing auto \
  --normalise-chord auto
```

`--normalise-chord yes` 强制平移、旋转和缩放到 `[0,0]`—`[1,0]`；
`--normalise-chord no` 保留输入坐标；默认 `auto` 会检查推断的弦长、方向和
前/尾缘横坐标，已经位于单位弦坐标系时不重复变换。

可显式提供原始坐标系中的前缘和虚拟尾缘：

```bash
python main.py \
  --airfoil "path/to/xxx.dat" \
  --normalise-chord yes \
  --leading-edge 0.0 0.0 \
  --trailing-edge 1.0 0.0
```


## 输入与输出

输入文件应为翼型轮廓坐标，第一行为名称，后续每行为一组 `x y` 坐标。轮廓点按上表面尾缘到前缘、再按下表面前缘到尾缘的顺序排列。

每个拟合结果会输出：

- `*_bp3333.dat`：重建后的翼型坐标；
- `*_bp3333.json`：12 个 BP3333 参数、MAE/RMS/最大绝对误差、实际优化器和根选择策略；
- `*_bp3333_thickness.png`：厚度与弯度分布对比图；
- `*_bp3333_airfoil.png`：原始翼型与拟合翼型对比图。

## 主要文件

- `model.py`：BP3333 参数定义、控制点计算和翼型生成；
- `fitting.py`：参数初值、约束和优化拟合；
- `geometry.py`：翼型坐标读取与几何量提取；
- `main.py`：命令行入口；
- `io.py` / `plotting.py`：结果保存与绘图。

## 参考文献

R. W. Derksen and T. Rogalsky, “Bezier-PARSEC: An optimized aerofoil parameterization for design,” *Advances in Engineering Software*, vol. 41, pp. 923-930, 2010. DOI: [10.1016/j.advengsoft.2010.05.002](https://doi.org/10.1016/j.advengsoft.2010.05.002)

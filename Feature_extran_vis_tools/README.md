## 医学影像边界特征提取与可视化工具

------

### 1. 项目概述

本项目旨在自动化地从医学影像（特别是伍德灯下的皮肤影像）中提取病灶边界的量化特征，并生成直观的可视化高亮叠加图。其核心功能是分析病灶边界的清晰度/模糊度，将其转化为数值指标，并以热力图的形式在原图上标注出来，辅助临床诊断和研究。

项目采用了高度可配置的批处理流程，允许研究人员使用不同的参数组合对整个数据集进行分析，并引入了“回退策略”，确保了处理的稳健性。即使在某些参数对特定影像无效时，也能尝试使用备用参数完成处理。

------

### 2. 项目结构与文件说明

```
.
├── feature_extractor.py          # 核心算法：单张影像的特征提取
├── batch_processor.py            # V1版本：对多种参数配置进行独立批处理
├── batch_processor_V2.py         # V2版本：带有回退策略的优化批处理程序 (推荐使用)
├── create_highlight_overlays.py  # 可视化工具：生成高亮叠加图
└── README.md                     # 本说明文件
```

#### `feature_extractor.py`

这是整个项目的核心算法模块。它包含一个主要的函数 `extract_woodlamp_edge_features`，其功能是：

- **输入**: 一张影像的路径和一系列图像处理参数（如对比度、二值化阈值、形态学内核大小等）。
- **处理流程**:
  - **影像增强**: 在 HSV 色彩空间中，对 V 通道（亮度）进行增强，以凸显病灶区域。
  - **病灶分割**: 使用自适应高斯阈值法（`cv2.adaptiveThreshold`）对增强后的灰度图进行二值化，以分割出病灶区域。
  - **形态学处理**: 通过开运算和闭运算去除噪点，平滑轮廓。
  - **轮廓筛选**: 根据面积滤除过小的、可能是噪点的轮廓。
  - **边界梯度量化**:
    - 在原始灰度图上计算梯度幅度图（使用 Sobel 算子）。
    - 提取分割出的病灶轮廓（Canny 边缘）。
    - 量化轮廓位置上的梯度值，计算其均值、标准差、最大值等，作为边界清晰度的指标。
- **输出**: 一个包含 `features`（量化指标字典）和 `gradient_map`（梯度幅度图）的字典。如果无法找到有效病灶，则返回 `None`。

#### `batch_processor_V2.py` (推荐使用的批处理程序)

此脚本用于对整个数据集进行自动化特征提取。它引入了**回退策略（Fallback Strategy）**，极大提升了处理成功率。

- **主要功能**:
  - 遍历指定目录下的所有影像。
  - 使用一组主要的、可能比较“激进”或“敏感”的参数 (`config_09_sensitive_and_strong_smoothing`) 进行首次尝试。
  - 如果首次尝试失败（即 `feature_extractor` 返回 `None`），它会自动使用一组更“保守”、更稳健的备用参数 (`config_01_baseline_c10_k5`) 进行第二次尝试。
  - 在特征 JSON 文件中记录下每张图片最终是使用哪个参数配置成功的。
- **输出**:
  - 一个名为 `features.json` 的文件，其中包含数据集中每张成功处理影像的边界特征指标。
  - 一系列与原图结构对应的 `png` 格式梯度图，这些图将用于后续的可视化。

#### `batch_processor.py` (V1版本)

这是批处理程序的初版。它不包含回退策略，而是独立地为每一组在 `PARAMETER_CONFIGS` 中定义的参数配置运行一次完整的数据集处理，并将结果保存在以配置名命名的独立子文件夹中。

- **适用场景**: 当您需要严格比较不同参数组合对整个数据集的影响时，此版本非常有用。

#### `create_highlight_overlays.py`

此脚本是可视化工具，用于将 `batch_processor_V2.py` 生成的抽象数据转化为直观的图像。

- **主要功能**:
  - 读取原始影像和对应的梯度图。
  - 使用一个高亮阈值 (`HIGHLIGHT_THRESHOLD`)，只将梯度值（即边界变化显著）超过该阈值的区域进行高亮。
  - 根据影像类型（临床或伍德灯）应用不同的颜色映射（`cv2.COLORMAP_COOL` 或 `cv2.COLORMAP_HOT`）生成热力图。
  - 使用蒙版技术，将生成的热力图仅叠加在梯度显著的边界区域上，而影像的其余部分保持原样。
- **输出**: 在指定输出目录下生成与原图结构相同的高亮叠加图。

------

### 3. 使用流程

#### 步骤 1: 准备数据

请按照以下建议的目录结构组织您的文件：

```
/path/to/your/project/
├── data/
│   └── processed/      <-- 存放原始影像的根目录
│       ├── patient_01/
│       │   ├── clinical_image.jpg
│       │   └── wood_lamp_image.jpg
│       └── ...
├── output/
│   ├── edge_features/  <-- 特征和梯度图的输出目录
│   └── overlays/       <-- 高亮叠加图的输出目录
└── src/
    ├── feature_extractor.py
    ├── batch_processor_V2.py
    └── create_highlight_overlays.py
```

#### 步骤 2: 配置路径

在运行脚本前，请打开 `batch_processor_V2.py` 和 `create_highlight_overlays.py`，修改文件开头的路径配置变量：

- **在 `batch_processor_V2.py` 中**:
  - `SOURCE_ROOT`: 设置为您的原始影像根目录 (例如: `/path/to/your/project/data/processed`)。
  - `OUTPUT_ROOT`: 设置为特征和梯度图的输出目录 (例如: `/path/to/your/project/output/edge_features`)。
- **在 `create_highlight_overlays.py` 中**:
  - `ORIGINAL_IMAGES_ROOT`: 应与 `batch_processor_V2.py` 的 `SOURCE_ROOT` 相同。
  - `FEATURE_MAPS_ROOT`: 应与 `batch_processor_V2.py` 的 `OUTPUT_ROOT` 相同。
  - `OVERLAY_OUTPUT_ROOT`: 设置为您希望保存高亮叠加图的目录 (例如: `/path/to/your/project/output/overlays`)。

#### 步骤 3: 运行特征提取

打开终端，导航到 `src` 目录，运行批处理脚本：

Bash

```
python batch_processor_V2.py
```

程序会显示进度条。完成后，您会在 `output/edge_features` 目录下看到一个 `features.json` 文件和所有成功处理的影像对应的梯度图（`.png`）。

#### 4. 生成可视化叠加图

特征提取完成后，运行可视化脚本：

Bash

```
python create_highlight_overlays.py
```

程序同样会显示进度条。完成后，您可以在 `output/overlays` 目录下找到所有带有边界高亮的影像。

### 4. 依赖库

本项目需要以下 Python 库：

- `opencv-python`
- `numpy`
- `matplotlib` (主要用于 `feature_extractor` 的调试，在批处理中非必需)
- `tqdm` (用于显示进度条)

您可以使用 `pip` 进行安装：

Bash

```
pip install opencv-python numpy matplotlib tqdm
```
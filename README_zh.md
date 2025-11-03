# hydromodel

[![image](https://img.shields.io/pypi/v/hydromodel.svg)](https://pypi.python.org/pypi/hydromodel)
[![image](https://img.shields.io/conda/vn/conda-forge/hydromodel.svg)](https://anaconda.org/conda-forge/hydromodel)
[![image](https://pyup.io/repos/github/OuyangWenyu/hydromodel/shield.svg)](https://pyup.io/repos/github/OuyangWenyu/hydromodel)

**轻量级水文模型率定和评估Python包，专注于新安江（XAJ）模型。**

- 开源协议: GNU General Public License v3
- 文档: https://OuyangWenyu.github.io/hydromodel

## hydromodel 是什么

`hydromodel` 是概念性水文模型的 Python 实现，专注于**新安江（XAJ）模型** - 中国及亚洲地区应用最广泛的降雨-径流模型之一。

**核心特性：**
- **XAJ 模型变体**: 标准 XAJ 和优化版本（xaj_mz 带 Muskingum 汇流）
- **多种率定算法**: SCE-UA、遗传算法和 scipy 优化器
- **全面的评估指标**: NSE、KGE、RMSE、PBIAS 等
- **统一的 API**: 率定和评估的一致接口
- **灵活的数据集成**: 通过 [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) 无缝支持 CAMELS 数据集，通过 [hydrodatasource](https://github.com/OuyangWenyu/hydrodatasource) 支持自定义数据
- **基于配置的工作流**: YAML 配置确保可重复性

## 为什么选择 hydromodel？

**对于研究者：**
- 经过实战检验的 XAJ 实现，已用于发表的研究
- 基于配置的工作流确保可重复性
- 易于扩展新模型或率定算法
- 轻量快速 - 非常适合参数敏感性研究

**对于实践者：**
- 简单的 YAML 配置，最少编码
- 高效处理多流域率定
- 集成全球 CAMELS 数据集（11 个变体）
- 清晰的文档和示例

**与其他包相比：**
- **vs. SWAT/VIC**: 更轻量，Python 原生，迭代更快
- **vs. pySTREPS**: 专注于概念性降雨-径流模型
- **vs. 自定义脚本**: 经过良好测试且具有统一接口

## 安装

### 普通用户

```bash
pip install hydromodel hydrodataset
```

或使用 `uv`（更快）：

```bash
uv pip install hydromodel hydrodataset
```

### 开发者

```bash
git clone https://github.com/OuyangWenyu/hydromodel.git
cd hydromodel
uv sync --all-extras
```

### 配置

#### 选项 1: 使用默认路径（推荐快速开始）

无需配置！`hydromodel` 自动使用默认路径：

**默认数据目录：**
- **Windows:** `C:\Users\YourUsername\hydromodel_data\`
- **macOS/Linux:** `~/hydromodel_data/`

默认结构（aqua_fetch 会自动创建大写的数据集目录）：
```
~/hydromodel_data/
├── datasets-origin/
│   ├── CAMELS_US/        # CAMELS US 数据集（由 aqua_fetch 创建）
│   ├── CAMELS_AUS/       # CAMELS Australia 数据集（如果使用）
│   └── ...               # 其他数据集
├── basins-origin/        # 你的自定义流域数据
└── ...
```

#### 选项 2: 自定义路径（高级用户）

创建 `~/hydro_setting.yml` 指定自定义路径：

```yaml
local_data_path:
  root: 'D:/data'
  datasets-origin: 'D:/data'             # CAMELS 数据集（aqua_fetch 会自动添加 CAMELS_US）
  basins-origin: 'D:/data/my_basins'     # 自定义数据
```

**重要说明**：对于 CAMELS 数据集，只需提供 `datasets-origin` 目录。系统会自动添加大写的数据集目录名（如 `CAMELS_US`、`CAMELS_AUS`）。例如数据在 `D:/data/CAMELS_US/`，则设置 `datasets-origin: 'D:/data'`。

## 使用方法

### 1. 数据准备

**使用 CAMELS 数据集 (hydrodataset)：**

```bash
pip install hydrodataset
```

```python
from hydrodataset.camels_us import CamelsUs

# 自动下载（如不存在）。提供 datasets-origin 目录（例如 "D:/data"）
# aqua_fetch 会自动添加数据集名称，创建 "D:/data/CAMELS_US/"
ds = CamelsUs(data_path)
basin_ids = ds.read_object_ids()  # 获取流域 ID
```

**注意：** 首次下载可能需要一段时间，完整的 CAMELS 数据集约为 70GB。

**可用数据集：** camels_us, camels_aus, camels_br, camels_ch, camels_cl, camels_gb, camels_de, camels_dk, camels_fr, camels_nz, camels_se

**使用自定义数据 (hydrodatasource)：**

对于你自己的数据，使用 `selfmadehydrodataset` 格式：

```bash
pip install hydrodatasource
```

**数据结构：**
```
my_basin_data/
├── attributes/
│   └── attributes.csv              # 流域元数据（必需）
├── timeseries/
│   ├── 1D/                         # 日尺度时间序列
│   │   ├── basin_001.csv          # 每个流域一个文件
│   │   ├── basin_002.csv
│   │   └── ...
│   └── 1D_units_info.json          # 变量单位（必需）
```

**必需文件：**
- `attributes.csv`: 必须有 `basin_id` 和 `area`（km²）列
- `{basin_id}.csv`: 时间序列，包含 `time` 列 + 变量（`prcp`、`PET`、`streamflow`）
- `{time_scale}_units_info.json`: 每个变量的单位（例如 `{"prcp": "mm/day"}`）

**在 hydromodel 中使用：**
```python
config = {
    "data_cfgs": {
        "data_source_type": "selfmadehydrodataset",  # 自定义数据使用此项
        "data_source_path": "D:/my_basin_data",      # 数据路径
        "basin_ids": ["basin_001"],
        ...
    }
}
```

详细格式规范和示例，请参见：
- [数据准备指南](docs/data_guide.md) - CAMELS 和自定义数据的完整指南
- [hydrodatasource 文档](https://github.com/OuyangWenyu/hydrodatasource) - 源包

### 2. 快速开始：率定、评估、模拟和可视化

**方式 1: 使用命令行脚本（推荐初学者）**

我们提供了现成的脚本用于模型率定、评估、模拟和可视化：

```bash
# 1. 率定
python scripts/run_xaj_calibration.py --config configs/example_config.yaml

# 2. 在测试期评估
python scripts/run_xaj_evaluate.py --calibration-dir results/xaj_mz_SCE_UA --eval-period test

# 3. 使用自定义参数模拟（无需率定！）
python scripts/run_xaj_simulate.py \
    --config configs/example_simulate_config.yaml \
    --param-file configs/example_xaj_params.yaml \
    --plot

# 4. 可视化
python scripts/visualize.py --eval-dir results/xaj_mz_SCE_UA/evaluation_test
```

编辑 `configs/example_config.yaml` 来自定义你的流域 ID、时间段和参数。

**方式 2: 使用 Python API（高级用户）**

```python
from hydromodel.trainers.unified_calibrate import calibrate
from hydromodel.trainers.unified_evaluate import evaluate

config = {
    "data_cfgs": {
        "data_source_type": "camels_us",
        "basin_ids": ["01013500"],
        "train_period": ["1985-10-01", "1995-09-30"],
        "test_period": ["2005-10-01", "2014-09-30"],
        "warmup_length": 365,
        "variables": ["precipitation", "potential_evapotranspiration", "streamflow"]
    },
    "model_cfgs": {
        "model_name": "xaj_mz",
    },
    "training_cfgs": {
        "algorithm_name": "SCE_UA",
        "algorithm_params": {"rep": 5000, "ngs": 1000},
        "loss_config": {"type": "time_series", "obj_func": "RMSE"},
        "output_dir": "results",
        "experiment_name": "my_experiment",
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE"],
    },
}

results = calibrate(config)  # 率定
evaluate(config, param_dir="results/my_experiment", eval_period="test")  # 评估
```

结果保存在 `results/` 目录中。

## 核心 API

### 配置结构

统一 API 使用包含四个主要部分的配置字典：

```python
config = {
    "data_cfgs": {
        "data_source_type": "camels_us",       # 数据集类型
        "basin_ids": ["01013500"],             # 要率定的流域 ID
        "train_period": ["1990-10-01", "2000-09-30"],
        "test_period": ["2000-10-01", "2010-09-30"],
        "warmup_length": 365,                  # 预热天数
        "variables": ["precipitation", "potential_evapotranspiration", "streamflow"],
    },
    "model_cfgs": {
        "model_name": "xaj_mz",                # 模型变体
        "model_params": {
            "source_type": "sources",
            "source_book": "HF",
        },
    },
    "training_cfgs": {
        "algorithm_name": "SCE_UA",            # SCE_UA、GA 或 scipy
        "algorithm_params": {
            "rep": 1000,                      # 迭代次数
            "ngs": 1000,                        # 复形数（SCE_UA）
        },
        "loss_config": {
            "type": "time_series",
            "obj_func": "RMSE",                # RMSE、NSE 或 KGE
        },
        "output_dir": "results",
        "experiment_name": "my_exp",
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE", "PBIAS"],
    },
}
```

### 率定 API

```python
from hydromodel.trainers.unified_calibrate import calibrate

results = calibrate(config)
```

**输出：** 率定结果保存到 `{output_dir}/{experiment_name}/`

**可用算法：**
- `SCE_UA`: 混洗复形演化算法（推荐）
- `GA`: 遗传算法
- `scipy`: scipy.optimize 方法

### 评估 API

```python
from hydromodel.trainers.unified_evaluate import evaluate

# 在测试期评估
test_results = evaluate(config, param_dir="results/my_exp", eval_period="test")

# 在训练期评估
train_results = evaluate(config, param_dir="results/my_exp", eval_period="train")

# 在自定义期评估
custom_results = evaluate(
    config,
    param_dir="results/my_exp",
    eval_period="custom",
    custom_period=["2010-10-01", "2015-09-30"]
)
```

**输出：** 评估结果在 `{param_dir}/evaluation_{period}/`
- `basins_metrics.csv` - 性能指标
- `basins_denorm_params.csv` - 率定参数
- `xaj_mz_evaluation_results.nc` - 完整模拟结果

**可用指标：** NSE, KGE, RMSE, PBIAS, FHV, FLV, FMS

### 模拟 API

**重要说明：模拟功能无需事先率定！**

`UnifiedSimulator` 提供了灵活的接口，可以使用任意参数值运行模型模拟：

```python
from hydromodel.trainers.unified_simulate import UnifiedSimulator
from hydromodel.datasets.unified_data_loader import UnifiedDataLoader

# 加载数据
data_loader = UnifiedDataLoader(config["data_cfgs"])
p_and_e, qobs = data_loader.load_data()

# 定义参数（可来自率定结果、文献或自定义值）
parameters = {
    "K": 0.75, "B": 0.25, "IM": 0.06,
    "UM": 18.0, "LM": 80.0, "DM": 95.0,
    # ... 其他参数
}

# 创建模拟器
model_config = {
    "model_name": "xaj_mz",
    "parameters": parameters
}
simulator = UnifiedSimulator(model_config, basin_config)

# 运行模拟
results = simulator.simulate(
    inputs=p_and_e,
    qobs=qobs,
    warmup_length=365
)

# 提取结果
qsim = results["qsim"]  # 模拟径流
```

**命令行使用：**

```bash
# 使用自定义参数（适用于任意参数值）
python scripts/run_xaj_simulate.py \
    --config configs/example_simulate_config.yaml \
    --param-file configs/example_xaj_params.yaml \
    --output simulation_results.csv \
    --plot

# 使用 SCE-UA 率定结果（CSV 格式）
python scripts/run_xaj_simulate.py \
    --param-file results/xaj_mz_SCE_UA/01013500_sceua.csv \
    --plot
```

**使用场景：**
- 参数敏感性分析
- 模型比较
- 使用自定义参数进行情景测试
- 验证文献参数

详细的 API 文档和高级用法，请参见 [使用指南 - 模型模拟](docs/usage.md#model-simulation)。

## 项目结构

```
hydromodel/
├── hydromodel/
│   ├── models/                      # 模型实现
│   │   ├── xaj.py                   # 标准 XAJ 模型
│   │   ├── gr4j.py                  # GR4J 模型
│   │   └── ...
│   ├── trainers/                    # 率定、评估和模拟
│   │   ├── unified_calibrate.py     # 率定 API
│   │   ├── unified_evaluate.py      # 评估 API
│   │   └── unified_simulate.py      # 模拟 API
│   └── datasets/                    # 数据预处理
│       ├── unified_data_loader.py   # 数据加载器
│       └── ...
├── scripts/                         # 示例脚本
│   ├── run_xaj_calibration.py       # 率定脚本
│   ├── run_xaj_evaluate.py          # 评估脚本
│   ├── run_xaj_simulate.py          # 模拟脚本
│   └── visualize.py                 # 可视化脚本
├── configs/                         # 配置文件
└── docs/                            # 文档
```

## 文档

- **快速开始**: [docs/quickstart.md](docs/quickstart.md)
- **使用指南**: [docs/usage.md](docs/usage.md)
- **API 参考**: https://OuyangWenyu.github.io/hydromodel

## 参考文献

**XAJ 模型：**
- Zhao, R.J., 1992. The Xinanjiang model applied in China. Journal of Hydrology, 135(1-4), pp.371-381.

**率定算法：**
- Duan, Q., et al., 1992. Effective and efficient global optimization for conceptual rainfall-runoff models. Water Resources Research, 28(4), pp.1015-1031. (SCE-UA)

**相关项目：**
- [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) - CAMELS 和其他数据集
- [hydrodatasource](https://github.com/OuyangWenyu/hydrodatasource) - 数据准备工具
- [torchhydro](https://github.com/OuyangWenyu/torchhydro) - 基于 PyTorch 的水文模型

## 引用

如果你在研究中使用 hydromodel，请引用：

```bibtex
@software{hydromodel,
  author = {Ouyang, Wenyu},
  title = {hydromodel: A Python Package for Hydrological Model Calibration},
  year = {2025},
  url = {https://github.com/OuyangWenyu/hydromodel}
}
```

## 贡献

欢迎贡献！对于重大更改，请先开 issue 讨论。

```bash
git clone https://github.com/OuyangWenyu/hydromodel.git
cd hydromodel
uv sync --all-extras
pytest tests/
```

## 许可证

GNU General Public License v3.0 - 详见 [LICENSE](LICENSE) 文件。

## 联系方式

- **作者**: Wenyu Ouyang
- **邮箱**: wenyuouyang@outlook.com
- **GitHub**: https://github.com/OuyangWenyu/hydromodel
- **问题**: https://github.com/OuyangWenyu/hydromodel/issues

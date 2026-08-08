# hydromodel

[![image](https://img.shields.io/pypi/v/hydromodel.svg)](https://pypi.python.org/pypi/hydromodel)
[![image](https://img.shields.io/conda/vn/conda-forge/hydromodel.svg)](https://anaconda.org/conda-forge/hydromodel)
[![image](https://pyup.io/repos/github/OuyangWenyu/hydromodel/shield.svg)](https://pyup.io/repos/github/OuyangWenyu/hydromodel)

**轻量级水文模型率定、评估与模拟 Python 包。**

- 开源协议：GNU General Public License v3
- 文档：https://OuyangWenyu.github.io/hydromodel

## hydromodel 是什么

`hydromodel` 是概念性水文模型的 Python 实现，专注于**新安江（XAJ）模型**——中国及亚洲地区应用最广泛的降雨-径流模型之一。

**已注册模型（`MODEL_DICT`）：**
- **XAJ 系列**：`xaj`、`xaj_mz`（带 MizuRoute 风格汇流）、`xaj_slw`（松辽变体）、`semi_xaj`
- **GR 系列**：`gr1a`、`gr2m`、`gr3j`、`gr4j`、`gr5j`、`gr6j`
- **其他**：`hymod`、`dhf`、`unit_hydrograph`、`categorized_unit_hydrograph`

**核心特性：**
- **多种率定算法**：
  - **SCE-UA**：基于 spotpy 的洗牌复形演化算法
  - **GA**：基于 DEAP 的遗传算法
  - **scipy**：L-BFGS-B、SLSQP 等梯度优化方法
- **多流域支持**：可对多个流域进行率定与评估（逐流域进行）
- **统一结果格式**：所有算法均保存为标准化的 JSON + CSV 格式
- **全面的评估指标**：NSE、KGE、RMSE、PBIAS、FHV、FLV 等
- **统一 API**：率定、评估、模拟使用一致接口
- **本地与云端数据**：支持从本地磁盘或 OSS/S3 读取公共数据集（CAMELS 系列等）与自定义数据（`source: local` / `source: cloud`）
- **基于配置的工作流**：YAML 配置保证可复现性
- **参数契约**：对 `param_range_file` 严格校验（文件缺失、参数未知/缺失、范围非法均快速失败）

## 为什么选择 hydromodel？

**对于研究者：**
- 经过实战检验的 XAJ 实现，已用于公开发表的研究
- 基于配置的工作流保证可复现性
- 易于扩展新模型或率定算法

**对于实践者：**
- 简单的 YAML 配置，几乎无需编码
- 支持多流域率定
- 集成 27 个注册的公共数据集以及自定义流域数据
- 文档与示例清晰

## 安装

### 普通用户

```bash
pip install hydromodel hydrodataset hydrodatasource
```

或使用 `uv`（更快）：

```bash
uv pip install hydromodel hydrodataset hydrodatasource
```

### 开发环境

开发者建议使用 `uv` 管理环境，因为本项目有本地依赖（如 `hydroutils`、`hydrodataset`、`hydrodatasource`）。

1. **克隆仓库：**
   ```bash
   git clone https://github.com/OuyangWenyu/hydromodel.git
   cd hydromodel
   ```

2. **使用 `uv` 同步环境：**
   该命令会安装所有依赖，包括 `[tool.uv.sources]` 中声明的本地可编辑包。
   ```bash
   uv sync --all-extras
   ```

## 配置

`hydromodel` 不再自行解析数据路径。数据加载委托给 `hydrodatasource` / `hydrodataset`
（见 `hydromodel/datasets/unified_data_loader.py`），由它们读取 `hydro_setting.yml` 中的存储配置。

### 存储配置

存储配置写在以下两个 YAML 文件之一的 `storage:` 块中：

1. `~/hydro_setting.yml` - 用户级，跨项目共享
2. `{项目根目录}/.hydro_setting.yml` - 项目级；存在时逐键覆盖用户级配置

```yaml
storage:
  default_source: local          # 未显式指定 data_cfgs.source 时使用
  local:
    root: F:/data                # 数据集父目录（包含 CAMELS_US/ 等数据集文件夹）
  cache: D:/netcdf               # NetCDF/zarr 缓存目录
  s3:                            # 仅 source: cloud 时需要
    bucket: hydrodataset
    prefix: ''
    access_key_id: 你的访问密钥
    secret_access_key: 你的私密密钥
    endpoint_url: https://oss-cn-beijing.aliyuncs.com
```

**重要说明：**
- `storage.local.root` 是**父目录**，其中包含各数据集文件夹（如 `F:/data/CAMELS_US/`）。数据集读取器会自动追加数据集文件夹名。新解析器**不再读取**旧的 `local_data_path` 块。
- 设置 `data_cfgs.source: cloud`（或 `storage.default_source: cloud`）可从 OSS/S3 读取。云端读取使用 `s3://<bucket>/zarr/` 下的 zarr 缓存；缓存缺失时会从原始文件生成（需要写权限）。
- `source` 是硬性选择：所选后端不可用时直接报错，不会静默回退。

### 数据集注册表

数据集 id 通过分层注册表解析：先查内置默认（在 `hydrodataset`/`hydrodatasource` 中），再查项目级可选的 `configs/datasets.yml` 覆盖。参见下文[支持的数据集](#支持的数据集)。

## 使用方法

### 1. 数据准备

**公共数据集（hydrodataset）：**

```python
from hydrodatasource.configs.data_resolver import open_dataset

ds = open_dataset("camels_us", source="local")   # 或 source="cloud"
basin_ids = ds.read_object_ids()                 # 例如 CAMELS-US 的 671 个流域
```

首次下载可能耗时较长（CAMELS-US 含压缩与解压文件约 70GB）。

**自定义数据（hydrodatasource）：**

可以使用已注册的自定义数据集 id，或用 `uri` + `reader` 直接指向你的数据：

```python
config = {
    "data_cfgs": {
        "dataset": "songliao_event",   # 已注册的自定义数据集（洪水场次）
        # --- 或任意本地数据 ---
        # "dataset": "my_basin",
        # "uri": "D:/data/my_basins",   # 显式路径，绕过注册表
        # "reader": "selfmade",         # reader 别名（selfmade、floodevent、longterm 等）
        "source": "local",
        "basin_ids": ["songliao_21401550"],
        "variables": ["rain", "ES", "inflow", "flood_event"],
        "warmup_length": 30,
        "is_event_data": True,
    },
    ...
}
```

可用的 reader 别名：`floodevent`、`selfmade`、`longterm`、`forecast`、`station`、
`tghydro`、`gages`、`grdc`、`rainfall`、`crd`、`rsvrinflow`。

额外的 reader 参数（`dataset_name`、`time_unit`、`datasource_kwargs` 等）会透传给读取器构造函数。完整自定义数据示例见 `configs/example_config_selfmade.yaml`。

### 2. 快速开始：率定、评估、模拟与可视化

**方式 1：使用命令行脚本（推荐）**

```bash
# 1. 率定（默认保存配置文件）
uv run python scripts/run_xaj_calibration.py --config configs/example_config.yaml

# 2. 在测试期评估（使用上面率定生成的 calibration_results.json）
uv run python scripts/run_xaj_evaluate.py \
    --calibration-dir results/12025000/experiment \
    --eval-period test

# 3. 使用自定义参数模拟（无需率定）
uv run python scripts/run_xaj_simulate.py \
    --config configs/example_simulate_config.yaml \
    --param-file configs/example_xaj_params.yaml \
    --plot

# 4. 可视化（过程线、散点、FDC、月均图 -> eval_dir/figures）
uv run python scripts/visualize.py --eval-dir results/12025000/experiment/evaluation_test

# 指定流域 / 图类型
uv run python scripts/visualize.py \
    --eval-dir results/12025000/experiment/evaluation_test \
    --basins 12025000 --plot-types timeseries scatter
```

**配置文件：**
- `configs/example_config.yaml` - 连续时间序列数据（如 CAMELS 数据集）
- `configs/example_config_selfmade.yaml` - 自定义数据 / 洪水场次数据
- `configs/example_simulate_config.yaml` - 模拟配置
- `configs/example_xaj_params.yaml` - XAJ 参数值示例（仅用于模拟）

**方式 2：使用 Python API（高级用户）**

```python
from hydromodel.trainers.unified_calibrate import calibrate
from hydromodel.trainers.unified_evaluate import evaluate

config = {
    "data_cfgs": {
        "dataset": "camels_us",
        "source": "local",
        "basin_ids": ["01013500"],
        "warmup_length": 365,
        "variables": ["precipitation", "potential_evapotranspiration", "streamflow"],
        "train_period": ["1985-10-01", "1995-09-30"],
        "test_period": ["2005-10-01", "2014-09-30"],
    },
    "model_cfgs": {
        "name": "xaj_mz",
        "params": {
            "source_type": "sources",
            "source_book": "HF",
            "kernel_size": 15,
        },
    },
    "training_cfgs": {
        "algorithm": "SCE_UA",
        "SCE_UA": {"rep": 1000, "ngs": 1000, "random_seed": 1234},
        "loss": "RMSE",
        "output_dir": "results",
        "experiment_name": "my_exp",
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE", "PBIAS"],
    },
}

results = calibrate(config)                          # 率定
evaluate(config, param_dir="results/my_exp", eval_period="test")  # 评估
```

结果保存在 `{training_cfgs.output_dir}/{training_cfgs.experiment_name}/` 下。

## 核心 API

### 配置结构

统一 API 使用包含四个部分的配置字典（`data_cfgs`、`model_cfgs`、`training_cfgs`、`evaluation_cfgs`）：

```python
config = {
    "data_cfgs": {
        "dataset": "camels_us",        # 注册表中的数据集 id（必填）
        "source": "local",             # "local" 或 "cloud"
        "basin_ids": ["01013500"],     # 待率定的流域
        "warmup_length": 365,          # 预热步数
        "variables": ["precipitation", "potential_evapotranspiration", "streamflow"],
        "train_period": ["1985-10-01", "1995-09-30"],
        "test_period": ["2005-10-01", "2014-09-30"],
    },
    "model_cfgs": {
        "name": "xaj_mz",              # MODEL_DICT 中的模型名
        "params": {                    # 模型专属配置
            "source_type": "sources",
            "source_book": "HF",
            "kernel_size": 15,
        },
        "output_variable": "qsim",     # 可选
    },
    "training_cfgs": {
        "algorithm": "SCE_UA",         # SCE_UA、GA 或 scipy
        # 算法专属超参数，以算法名作为键：
        "SCE_UA": {
            "rep": 1000,
            "ngs": 1000,
            "kstop": 500,
            "peps": 0.1,
            "pcento": 0.1,
            "random_seed": 1234,
        },
        "GA": {
            "pop_size": 40,
            "n_generations": 20,
            "cx_prob": 0.7,
            "mut_prob": 0.2,
            "random_seed": 1234,
        },
        "scipy": {
            "method": "SLSQP",
            "max_iterations": 500,
        },
        "loss": "RMSE",                # RMSE、NSE、KGE、LOGNSE 等
        "output_dir": "results",
        "experiment_name": "my_exp",
        "param_range_file": None,      # 可选的自定义参数范围
        "save_config": True,
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE", "PBIAS"],
        "save_results": True,
        "plot_results": True,
    },
}
```

**说明：**
- `data_cfgs.dataset` **必填**。自定义数据请添加 `uri`（显式路径）和/或 `reader`。
- `training_cfgs.loss`（字符串）内部会被包装成 `loss_config`；也接受完整的 `loss_config` 字典。
- 优化器始终**最小化**目标函数。用户目标 `NSE`、`KGE`、`LOGNSE` 内部会映射为取负的目标（如 `KGE -> neg_kge`）。
- 显式指定 `param_range_file` 时严格校验：文件缺失、参数未知、参数缺失或 `[min, max]` 范围非法都会快速失败。未指定时使用内置 `MODEL_PARAM_DICT` 范围。

### 率定 API

```python
from hydromodel.trainers.unified_calibrate import calibrate

results = calibrate(config)
```

**保存的文件**（位于 `{output_dir}/{experiment_name}/`）：
```
calibration_results.json          # 所有流域的最优参数（统一格式）
{basin_id}_sceua.csv              # SCE-UA 迭代历史（按算法）
{basin_id}_ga.csv                 # GA 代数历史
{basin_id}_scipy.csv              # scipy 迭代历史
calibration_config.yaml           # 使用的配置（save_config=True 时保存）
param_range.yaml                  # 解析后的参数范围（save_config=True 时保存）
```

**说明：**
- `calibration_results.json` 始终保存，是评估的主要输入。
- `best_params` 保持归一化 `[0,1]`；物理参数值请使用 `best_params_denormalized`。
- JSON 中的 `param_range_source` / `param_range_source_path` 记录参数范围来源。

### 评估 API

```python
from hydromodel.trainers.unified_evaluate import evaluate

test_results = evaluate(config, param_dir="results/my_exp", eval_period="test")
train_results = evaluate(config, param_dir="results/my_exp", eval_period="train")
```

**输出：** `{param_dir}/evaluation_{period}/`
- `basins_metrics.csv` - 性能指标
- `basins_norm_params.csv` / `basins_denorm_params.csv` - 率定参数
- `<model>_evaluation_results.nc` - 完整模拟结果（NetCDF）
- `evaluation_info.yaml` - 评估元信息

**参数加载优先级：** 优先 `calibration_results.json`，其次各算法遗留 CSV/txt 文件。

**可用指标：** NSE、KGE、RMSE、PBIAS、FHV、FLV、FMS 等。

### 模拟 API

模拟**无需**事先率定。`UnifiedSimulator` 可以用任意参数值运行模型：

```python
from hydromodel.datasets.unified_data_loader import UnifiedDataLoader
from hydromodel.trainers.unified_simulate import UnifiedSimulator

# 加载数据
data_loader = UnifiedDataLoader(config["data_cfgs"])
p_and_e, qobs = data_loader.load_data()

# 模型配置：model_name + model_params + 具体参数值
model_config = {
    "model_name": "xaj",
    "parameters": {"K": 0.75, "B": 0.25, "IM": 0.06, "UM": 18.0, ...},
}
simulator = UnifiedSimulator(model_config)   # basin_config 可选

results = simulator.simulate(inputs=p_and_e, qobs=qobs, warmup_length=365)
qsim = results["qsim"]   # 模拟径流
```

**命令行用法：**
```bash
# 自定义参数（YAML）
uv run python scripts/run_xaj_simulate.py \
    --config configs/example_simulate_config.yaml \
    --param-file configs/example_xaj_params.yaml \
    --output simulation_results.csv \
    --plot

# 使用 SCE-UA 率定结果（CSV 旧格式）
uv run python scripts/run_xaj_simulate.py \
    --param-file results/my_exp/01013500_sceua.csv \
    --plot
```

## 支持的数据集

权威的运行时注册表位于 `hydrodataset`（公共数据集）和 `hydrodatasource`（自定义数据集），不在 hydromodel 中。可以通过项目级 `configs/datasets.yml` 扩展或覆盖条目。

**公共数据集（27 个，via hydrodataset）：**
- **CAMELS 系列（16）**：`camels_us`、`camels_aus`、`camels_br`、`camels_ch`、`camels_cl`、`camels_col`、`camels_de`、`camels_dk`、`camels_fi`、`camels_pe`、`camels_fr`、`camels_gb`、`camels_ind`、`camels_lux`、`camels_nz`、`camels_se`
- **CAMELSH 系列（2）**：`camelsh`、`camelsh_kr`
- **CARAVAN 系列（3）**：`caravan`、`caravan_dk`、`grdc_caravan`
- **LamaH 系列（2）**：`lamah_ce`、`lamah_ice`
- **其他（4）**：`hysets`（加拿大）、`bull`（法国）、`estreams`（欧洲）、`simbi`（巴西）

**自定义数据集（via hydrodatasource）：**
- `songliao_event` - 松辽洪水场次数据集（已注册 id）
- 通过 `uri` + `reader` 别名读取任意本地/云端数据（`selfmade`、`floodevent`、`longterm`、`forecast`、`station`、`tghydro`、`gages`、`grdc`、`rainfall`、`crd`、`rsvrinflow`）

注意：hydromodel 的 `hydromodel/datasets/dataset_dict.py` 只是参考映射；其中列出的部分名称（如 `camels_deby`、`mopex`、`hype`）未在运行时注册表中启用，需要 `configs/datasets.yml` 条目或 `uri` 才能使用。

## 项目结构

```
hydromodel/
├── hydromodel/
│   ├── configs/                     # 统一配置管理与校验
│   │   └── config_manager.py
│   ├── models/                      # 模型实现（注册于 MODEL_DICT）
│   │   ├── xaj.py, xaj_slw.py, semi_xaj.py
│   │   ├── gr1a.py ... gr6j.py      # GR 系列
│   │   ├── hymod.py, dhf.py, unit_hydrograph.py
│   │   ├── model_dict.py            # MODEL_DICT / LOSS_DICT 注册表
│   │   └── model_config.py          # 参数契约与校验
│   ├── trainers/
│   │   ├── unified_calibrate.py     # 率定 API（SCE-UA / GA / scipy）
│   │   ├── unified_evaluate.py      # 评估 API
│   │   └── unified_simulate.py      # 模拟 API
│   ├── datasets/
│   │   ├── unified_data_loader.py   # 统一数据加载（委托给 open_dataset）
│   │   ├── dataset_dict.py          # 数据集参考映射
│   │   └── data_visualize.py        # 绘图函数
│   └── __init__.py                  # 顶层懒加载 API（list_models、describe_model 等）
├── scripts/                         # 命令行脚本
│   ├── run_xaj_calibration.py
│   ├── run_xaj_evaluate.py
│   ├── run_xaj_simulate.py
│   └── visualize.py
├── configs/                         # 示例 YAML 配置
├── test/                            # 测试（运行：pytest test/）
└── docs/                            # 文档
```

## 文档

- **快速开始**：[docs/quickstart.md](docs/quickstart.md)
- **使用指南**：[docs/usage.md](docs/usage.md)
- **数据指南**：[docs/data_guide.md](docs/data_guide.md)
- **数据路径解析（ADR 0001）**：[docs/adr/0001-unified-data-path-resolution.md](docs/adr/0001-unified-data-path-resolution.md)
- **API 参考**：https://OuyangWenyu.github.io/hydromodel

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

如果你在研究中使用了 hydromodel，请引用：

```bibtex
@software{hydromodel,
  author = {Ouyang, Wenyu},
  title = {hydromodel: A Python Package for Hydrological Model Calibration},
  year = {2025},
  url = {https://github.com/OuyangWenyu/hydromodel}
}
```

## 贡献

欢迎贡献！对于重大改动，请先开 issue 讨论。

```bash
git clone https://github.com/OuyangWenyu/hydromodel.git
cd hydromodel
uv sync --all-extras
pytest test/
```

## 许可证

GNU General Public License v3.0 - 详见 [LICENSE](LICENSE) 文件。

## 联系方式

- **作者**：Wenyu Ouyang
- **邮箱**：wenyuouyang@outlook.com
- **GitHub**：https://github.com/OuyangWenyu/hydromodel
- **问题**：https://github.com/OuyangWenyu/hydromodel/issues
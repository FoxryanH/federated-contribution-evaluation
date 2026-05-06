# 《面向数据交易的质量感知联邦学习贡献评价方法》实验代码

本仓库包含一组用于联邦学习贡献评估的实验代码与 JSON 结果数据，覆盖准确性、效率、攻击鲁棒性、适应性、参数敏感度和可解释性等分析场景。

仓库中的目录和文件路径已统一为英文，便于上传 GitHub 和跨平台使用。

## 目录结构

```text
.
+-- 1.accuracy_analysis/              # 准确性分析
+-- 2.efficiency_analysis/            # 效率分析
+-- 3.attack_robustness_analysis/     # 攻击鲁棒性分析
+-- 4.adaptability_analysis/          # 适应性分析
+-- 5.parameter_sensitivity_analysis/ # 参数敏感度分析
+-- 6.interpretability_analysis/      # 可解释性分析
+-- requirements.txt
`-- README.md
```

## 环境配置

建议使用 Python 3.10 或更高版本。

安装依赖：

```bash
pip install -r requirements.txt
```

如果需要使用 GPU，请根据本机 CUDA 版本安装对应的 PyTorch 版本，然后再安装 `requirements.txt` 中的其他依赖。

## 主要依赖

- `torch`：模型训练、联邦优化和张量计算
- `torchvision`：CIFAR-10 数据集加载和预处理
- `numpy`：数值计算和数据划分

本仓库已移除绘图代码，因此不再需要额外安装绘图库。

## 运行示例

运行 CIFAR-10 贡献评估实验：

```bash
python 1.accuracy_analysis/cifar10/federated_cifar10_mingaplr_compare.py --data-dir data --output-json results.json
```

运行受控准确性实验：

```bash
python 1.accuracy_analysis/controlled_experiments/federated_cifar10_mingaplr_S1_controlled_compare.py --data-dir data --output-json our-s1_results.json
```

运行效率分析实验：

```bash
python 2.efficiency_analysis/federated_cifar10_mingaplr_S1_time_analysis.py
```

运行攻击鲁棒性实验：

```bash
python 3.attack_robustness_analysis/cifar-10/federated_cifar10_hist_mingaplr_dirichlet05_window_noise_gaussian_noise_compare.py
```

生成可解释性分析表格数据：

```bash
python 6.interpretability_analysis/build_subjective_interpretability_table.py
```

大多数脚本都提供命令行参数。可以使用 `--help` 查看具体参数：

```bash
python 1.accuracy_analysis/cifar10/federated_cifar10_mingaplr_compare.py --help
```

## 数据说明

CIFAR-10 实验通过 `torchvision` 加载数据，可根据脚本参数将数据保存到指定的 `data` 目录。

Speech Commands 和 Yahoo Answers 相关实验需要准备对应的数据文件，具体路径可通过脚本中的命令行参数指定。

## 结果文件

仓库中保留的是实验代码、JSON 结果数据、`requirements.txt` 和本说明文档。PNG、PDF、CSV、HTML 等图表或表格产物已删除。

重新运行实验时，通常会通过 `--output-json` 或脚本中的其他输出参数生成新的 JSON 结果文件。可解释性脚本支持生成 CSV、Markdown 和 HTML 表格；如需保持仓库只包含代码和 JSON 数据，请不要提交这些派生表格文件。

## 注意事项

- 仓库路径已统一为英文，避免中文路径在 GitHub 或不同系统中产生兼容性问题。
- 已删除 `__pycache__` 等 Python 缓存目录。
- 已移除绘图代码和图表产物，仅保留 JSON 数据文件。
- 部分数据集脚本会引用本地模型定义，例如 `model.AudioCNN` 或 `model.TextCNN`。运行这些脚本前，请确认对应模型文件位于 Python 可导入路径中。

<div align="center">
    <h2>
        Time Travelling Pixels: Bitemporal Features Integration with Foundation Model for Remote Sensing Image Change Detection
    </h2>
</div>
<br>

[//]: # (<div align="center">)

[//]: # (  <img src="resources/RSPrompter.png" width="800"/>)

[//]: # (</div>)
<br>
<div align="center">
  <a href="https://kychen.me/TTP">
    <span style="font-size: 20px; ">项目主页</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/xxxx">
    <span style="font-size: 20px; ">arXiv</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/spaces/KyanChen/TTP">
    <span style="font-size: 20px; ">HFSpace</span>
  </a>
    &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="resources/ttp.pdf">
    <span style="font-size: 20px; ">PDF</span>
  </a>
</div>
<br>
<br>

[![GitHub stars](https://badgen.net/github/stars/KyanChen/TTP)](https://github.com/KyanChen/TTP)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-xxx-b31b1b.svg)](https://arxiv.org/abs/xxx)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/KyanChen/TTP)

<br>
<br>

<div align="center">

[English](README.md) | 简体中文

</div>


## 简介

本项目仓库是论文 [Time Travelling Pixels: Bitemporal Features Integration with Foundation Model for Remote Sensing Image Change Detection](https://arxiv.org/abs/xxxx) 的代码实现，基于 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 和 [Open-CD](https://github.com/likyoo/open-cd) 项目进行开发。

当前分支在PyTorch 2.x 和 CUDA 12.1 下测试通过，支持 Python 3.7+，能兼容绝大多数的 CUDA 版本。

如果你觉得本项目对你有帮助，请给我们一个 star ⭐️，你的支持是我们最大的动力。

<details open>
<summary>主要特性</summary>

- 与 MMSegmentation 高度保持一致的 API 接口及使用方法
- 开源了论文中的 TTP 模型
- 通过了 AMP 训练方式的测试
- 支持了多种数据集扩展

</details>

## 更新日志

🌟 **2023.12.23** 发布了 TTP 项目代码，完全与 MMSegmentation 保持一致的API接口及使用方法。

[//]: # (## TODO)


## 目录

- [简介](#简介)
- [更新日志](#更新日志)
- [目录](#目录)
- [安装](#安装)
- [数据集准备](#数据集准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [图像预测](#图像预测)
- [常见问题](#常见问题)
- [致谢](#致谢)
- [引用](#引用)
- [开源许可证](#开源许可证)
- [联系我们](#联系我们)

## 安装

### 依赖项

- Linux 或 Windows
- Python 3.7+，推荐使用 3.10
- PyTorch 2.0 或更高版本，推荐使用 2.1
- CUDA 11.7 或更高版本，推荐使用 12.1
- MMCV 2.0 或更高版本，推荐使用 2.1

### 环境安装

我们推荐使用 Miniconda 来进行安装，以下命令将会创建一个名为 `ttp` 的虚拟环境，并安装 PyTorch 和 MMCV。

注解：如果你对 PyTorch 有经验并且已经安装了它，你可以直接跳转到下一小节。否则，你可以按照下述步骤进行准备。

<details>

**步骤 0**：安装 [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)。

**步骤 1**：创建一个名为 `ttp` 的虚拟环境，并激活它。

```shell
conda create -n ttp python=3.10 -y
conda activate ttp
```

**步骤 2**：安装 [PyTorch](https://pytorch.org/get-started/locally/)。

Linux:
```shell
pip install torch torchvision torchaudio
```
Windows:
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**步骤 3**：安装 [MMCV](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)。

```shell
pip install -U openmim
mim install "mmcv>=2.0.0"
```

**步骤 4**：安装其他依赖项。

```shell
pip install -U wandb einops importlib peft scipy ftfy prettytable torchmetrics
```


</details>

### 安装 TTP

下载或克隆 TTP 仓库即可。

```shell
git clone git@github.com:KyanChen/TTP.git
cd TTP
```

## 数据集准备

<details>

### Levir-CD变化检测数据集

#### 数据下载

- 图片及标签下载地址： [Levir-CD](https://chenhao.in/LEVIR/)。


#### 组织方式

你也可以选择其他来源进行数据的下载，但是需要将数据集组织成如下的格式：

```
${DATASET_ROOT} # 数据集根目录，例如：/home/username/data/levir-cd
├── train
│   ├── A
│   ├── B
│   └── label
├── val
│   ├── A
│   ├── B
│   └── label
└── test
    ├── A
    ├── B
    └── label
```

注解：在项目文件夹中，我们提供了一个名为 `data` 的文件夹，其中包含了上述数据集的组织方式的示例。

### 其他数据集

如果你想使用其他数据集，可以参考 [MMSegmentation 文档](https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/2_dataset_prepare.html) 来进行数据集的准备。

</details>

## 模型训练

### TTP 模型

#### Config 文件及主要参数解析

我们提供了论文中使用的 TTP 模型的配置文件，你可以在 `configs/TTP` 文件夹中找到它们。Config 文件完全与 MMSegmentation 保持一致的 API 接口及使用方法。下面我们提供了一些主要参数的解析。如果你想了解更多参数的含义，可以参考 [MMSegmentation 文档](https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/1_config.html)。

<details>

**参数解析**：

- `work_dir`：模型训练的输出路径，一般不需要修改。
- `default_hooks-CheckpointHook`：模型训练过程中的检查点保存配置，一般不需要修改。
- `default_hooks-visualization`：模型训练过程中的可视化配置，**训练时注释，测试时取消注释**。
- `vis_backends-WandbVisBackend`：网络端可视化工具的配置，**打开注释后，需要在 `wandb` 官网上注册账号，可以在网络浏览器中查看训练过程中的可视化结果**。
- `sam_pretrain_ckpt_path`：MMPretrain 提供的 SAM 主干的检查点路径，参考[下载地址](https://github.com/open-mmlab/mmpretrain/tree/main/configs/sam)。
- `model-backbone-peft_cfg`：是否引入微调参数，一般不需要修改。
- `dataset_type`：数据集的类型，**需要根据数据集的类型进行修改**。
- `data_root`：数据集根目录，**修改为数据集根目录的绝对路径**。
- `batch_size_per_gpu`：单卡的 batch size，**需要根据显存大小进行修改**。
- `resume`: 是否断点续训，一般不需要修改。
- `load_from`：模型的预训练的检查点路径，一般不需要修改。
- `max_epochs`：最大训练轮数，一般不需要修改。

</details>


#### 单卡训练

```shell
python tools/train.py configs/TTP/xxx.py  # xxx.py 为你想要使用的配置文件
```

#### 多卡训练

```shell
sh ./tools/dist_train.sh configs/TTP/xxx.py ${GPU_NUM}  # xxx.py 为你想要使用的配置文件，GPU_NUM 为使用的 GPU 数量
```

### 其他实例分割模型

<details>

如果你想使用其他变化检测模型，可以参考 [Open-CD](https://github.com/likyoo/open-cd) 来进行模型的训练，也可以将其Config文件放入本项目的 `configs` 文件夹中，然后按照上述的方法进行训练。

</details>

## 模型测试

#### 单卡测试：

```shell
python tools/test.py configs/TTP/xxx.py ${CHECKPOINT_FILE}  # xxx.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件
```

#### 多卡测试：

```shell
sh ./tools/dist_test.sh configs/TTP/xxx.py ${CHECKPOINT_FILE} ${GPU_NUM}  # xxx.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件，GPU_NUM 为使用的 GPU 数量
```

**注解**：如果需要获取可视化结果，可以在 Config 文件中取消 `default_hooks-visualization` 的注释。


## 图像预测

#### 单张图像预测：

```shell
python demo/image_demo.py ${IMAGE_FILE}  configs/TTP/xxx.py --weights ${CHECKPOINT_FILE} --out-dir ${OUTPUT_DIR}  # IMAGE_FILE 为你想要预测的图像文件，xxx.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件，OUTPUT_DIR 为预测结果的输出路径
```

#### 多张图像预测：

```shell
python demo/image_demo.py ${IMAGE_DIR}  configs/TTP/xxx.py --weights ${CHECKPOINT_FILE} --out-dir ${OUTPUT_DIR}  # IMAGE_DIR 为你想要预测的图像文件夹，xxx.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件，OUTPUT_DIR 为预测结果的输出路径
```



## 常见问题

<details>

我们在这里列出了使用时的一些常见问题及其相应的解决方案。如果您发现有一些问题被遗漏，请随时提 PR 丰富这个列表。如果您无法在此获得帮助，请使用[issue](https://github.com/KyanChen/TTP/issues)来寻求帮助。请在模板中填写所有必填信息，这有助于我们更快定位问题。

### 1. 是否需要安装MMSegmentation，MMPretrain，MMDet，Open-CD？

我们建议您不要安装它们，因为我们已经对它们的代码进行了部分修改，如果您安装了它们，可能会导致代码运行出错。如果你出现了模块尚未被注册的错误，请检查：

- 是否安装了这些库，若有则卸载
- 是否在类名前加上了`@MODELS.register_module()`，若没有则加上
- 是否在`__init__.py`中加入了`from .xxx import xxx`，若没有则加上
- 是否在Config文件中加入了`custom_imports = dict(imports=['mmseg.ttp'], allow_failed_imports=False)`，若没有则加上


### 2. 关于资源消耗情况

这里我们列出了使用不同训练方法的资源消耗情况，供您参考。

| 模型名称 |  骨干网络类型  |  图像尺寸   |       GPU       | Batch Size | 加速策略 | 单卡显存占用  | 训练时间 |
|:----:|:--------:|:-------:|:---------------:|:----------:|:----:|:-------:|:----:|
| TTP  | ViT-L/16 | 512x512 | 4x RTX 4090 24G |     2      | FP32 |  14 GB  |  3H  |
| TTP  | ViT-L/16 | 512x512 | 4x RTX 4090 24G |     2      | FP16 |  12 GB  |  2H  |


### 4. dist_train.sh: Bad substitution的解决

如果您在运行`dist_train.sh`时出现了`Bad substitution`的错误，请使用`bash dist_train.sh`来运行脚本。


### 5. You should set `PYTHONPATH` to make `sys.path` include the directory which contains your custom module

请查看详细的报错信息，一般是某些依赖包没有安装，请使用`pip install`来安装依赖包。
</details>

## 致谢

本项目基于 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 和 [Open-CD](https://github.com/likyoo/open-cd) 项目进行开发，感谢 MMSegmentation 和 Open-CD 项目的开发者们。

## 引用

如果你在研究中使用了本项目的代码或者性能基准，请参考如下 bibtex 引用 TTP。

```
xxx
```

## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## 联系我们

如果有其他问题❓，请及时与我们联系 👬

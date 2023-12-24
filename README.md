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
    <span style="font-size: 20px; ">Project Page</span>
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

English | [简体中文](README_zh-CN.md)

</div>


## Introduction

The repository is the code implementation of the paper [Time Travelling Pixels: Bitemporal Features Integration with Foundation Model for Remote Sensing Image Change Detection](https://arxiv.org/abs/xxxx), based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [Open-CD](https://github.com/likyoo/open-cd) projects.

The current branch has been tested under PyTorch 2.x and CUDA 12.1, supports Python 3.7+, and is compatible with most CUDA versions.

If you find this project helpful, please give us a star ⭐️, your support is our greatest motivation.

<details open>
<summary>Main Features</summary>

- Consistent API interface and usage with MMSegmentation
- Open sourced the TTP model in the paper
- Tested with AMP training method
- Supports multiple dataset extensions

</details>

## Update Log

🌟 **2023.12.23** Released the TTP project code, which is completely consistent with the API interface and usage of MMSegmentation.


## Table of Contents

- [Introduction](#introduction)
- [Update Log](#update-log)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Model Testing](#model-testing)
- [Image Prediction](#image-prediction)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)
- [Contact Us](#contact-us)

## Installation

### Dependencies

- Linux or Windows
- Python 3.7+, recommended 3.10
- PyTorch 2.0 or higher, recommended 2.1
- CUDA 11.7 or higher, recommended 12.1
- MMCV 2.0 or higher, recommended 2.1

### Environment Installation

We recommend using Miniconda for installation. The following command will create a virtual environment named `ttp` and install PyTorch and MMCV.

Note: If you have experience with PyTorch and have already installed it, you can skip to the next section. Otherwise, you can follow these steps to prepare.

<details>

**Step 0**: Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).

**Step 1**: Create a virtual environment named `ttp` and activate it.

```shell
conda create -n ttp python=3.10 -y
conda activate ttp
```

**Step 2**: Install [PyTorch](https://pytorch.org/get-started/locally/).

Linux:
```shell
pip install torch torchvision torchaudio
```
Windows:
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Step 3**: Install [MMCV](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

```shell
pip install -U openmim
mim install "mmcv>=2.0.0"
```

**Step 4**: Install other dependencies.

```shell
pip install -U wandb einops importlib peft scipy ftfy prettytable torchmetrics
```


</details>

### Install TTP


Download or clone the TTP repository.

```shell
git clone git@github.com:KyanChen/TTP.git
cd TTP
```

## Dataset Preparation

<details>

### Levir-CD Change Detection Dataset

#### Dataset Download

- Image and label download address: [Levir-CD](https://chenhao.in/LEVIR/).

#### Organization Method

You can also choose other sources to download the data, but you need to organize the dataset in the following format:

```
${DATASET_ROOT} # Dataset root directory, for example: /home/username/data/levir-cd
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

Note: In the project folder, we provide a folder named `data`, which contains an example of the organization method of the above dataset.

### Other Datasets

If you want to use other datasets, you can refer to [MMSegmentation documentation](https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/2_dataset_prepare.html) to prepare the datasets.
</details>

## Model Training

### TTP Model

#### Config File and Main Parameter Parsing

We provide the configuration files of the TTP model used in the paper, which can be found in the `configs/TTP` folder. The Config file is completely consistent with the API interface and usage of MMSegmentation. Below we provide an analysis of some of the main parameters. If you want to know more about the meaning of the parameters, you can refer to [MMSegmentation documentation](https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/1_config.html).
<details>

**Parameter Parsing**:

- `work_dir`: The output path of the model training, which generally does not need to be modified.
- `default_hooks-CheckpointHook`: Checkpoint saving configuration during model training, which generally does not need to be modified.
- `default_hooks-visualization`: Visualization configuration during model training, **comment out during training and uncomment during testing**.
- `vis_backends-WandbVisBackend`: Configuration of network-side visualization tools, **after opening the comment, you need to register an account on the `wandb` official website, and you can view the visualization results during the training process in the network browser**.
- `sam_pretrain_ckpt_path`: The checkpoint path of the SAM backbone provided by MMPretrain, refer to [download address](https://github.com/open-mmlab/mmpretrain/tree/main/configs/sam).
- `model-backbone-peft_cfg`: Whether to introduce fine-tuning parameters, which generally does not need to be modified.
- `dataset_type`: The type of dataset, **needs to be modified according to the type of dataset**.
- `data_root`: Dataset root directory, **modify to the absolute path of the dataset root directory**.
- `batch_size_per_gpu`: The batch size of a single card, **needs to be modified according to the memory size**.
- `resume`: Whether to resume training, which generally does not need to be modified.
- `load_from`: The checkpoint path of the model's pre-training, which generally does not need to be modified.
- `max_epochs`: The maximum number of training rounds, which generally does not need to be modified.

</details>


#### Single Card Training

```shell
python tools/train.py configs/TTP/xxx.py  # xxx.py is the configuration file you want to use
```

#### Multi-card Training

```shell
sh ./tools/dist_train.sh configs/TTP/xxx.py ${GPU_NUM}  # xxx.py is the configuration file you want to use, GPU_NUM is the number of GPUs used
```

### Other Instance Segmentation Models

<details>

If you want to use other change detection models, you can refer to [Open-CD](https://github.com/likyoo/open-cd) to train the models, or you can put their Config files into the `configs` folder of this project, and then train them according to the above method.

</details>

## Model Testing

#### Single Card Testing:

```shell
python tools/test.py configs/TTP/xxx.py ${CHECKPOINT_FILE}  # xxx.py is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use
```

#### Multi-card Testing:

```shell
sh ./tools/dist_test.sh configs/TTP/xxx.py ${CHECKPOINT_FILE} ${GPU_NUM}  # xxx.py is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, GPU_NUM is the number of GPUs used
```

**Note**: If you need to get the visualization results, you can uncomment `default_hooks-visualization` in the Config file.


## Image Prediction

#### Single Image Prediction:

```shell
python demo/image_demo.py ${IMAGE_FILE}  configs/TTP/xxx.py --weights ${CHECKPOINT_FILE} --out-dir ${OUTPUT_DIR}  # IMAGE_FILE is the image file you want to predict, xxx.py is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, OUTPUT_DIR is the output path of the prediction result
```

#### Multiple Image Prediction:

```shell
python demo/image_demo.py ${IMAGE_DIR}  configs/TTP/xxx.py --weights ${CHECKPOINT_FILE} --out-dir ${OUTPUT_DIR}  # IMAGE_DIR is the image folder you want to predict, xxx.py is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, OUTPUT_DIR is the output path of the prediction result
```



## FAQ

<details>

We have listed some common problems and their corresponding solutions here. If you find that some problems are missing, please feel free to provide a PR to enrich this list. If you cannot get help here, please use [issue](https://github.com/KyanChen/TTP/issues) to seek help. Please fill in all the required information in the template, which will help us locate the problem faster.

### 1. Do I need to install MMSegmentation, MMPretrain, MMDet, Open-CD?

We recommend that you do not install them, because we have partially modified their code, which may cause errors in the code if you install them. If you get an error that the module has not been registered, please check:

- Whether these libraries are installed, if so, uninstall them
- Whether `@MODELS.register_module()` is added in front of the class name, if not, add it
- Whether `from .xxx import xxx` is added in `__init__.py`, if not, add it
- Whether `custom_imports = dict(imports=['mmseg.ttp'], allow_failed_imports=False)` is added in the Config file, if not, add it


### 2. About resource consumption

Here we list the resource consumption of using different training methods for your reference.


| Model Name |  Backbone Type  |  Image Size   |       GPU       | Batch Size | Acceleration Strategy | Single Card Memory Usage  | Training Time |
|:----:|:--------:|:-------:|:---------------:|:----------:|:----:|:-------:|:----:|
| TTP  | ViT-L/16 | 512x512 | 4x RTX 4090 24G |     2      | FP32 |  14 GB  |  3H  |
| TTP  | ViT-L/16 | 512x512 | 4x RTX 4090 24G |     2      | FP16 |  12 GB  |  2H  |




### 4. Solution to dist_train.sh: Bad substitution

If you get a `Bad substitution` error when running `dist_train.sh`, use `bash dist_train.sh` to run the script.


### 5. You should set `PYTHONPATH` to make `sys.path` include the directory which contains your custom module

Please check the detailed error message, generally some dependent packages are not installed, please use `pip install` to install the dependent packages.
</details>

## Acknowledgements

The repository is the code implementation of the paper [Time Travelling Pixels: Bitemporal Features Integration with Foundation Model for Remote Sensing Image Change Detection](https://arxiv.org/abs/xxxx), based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [Open-CD](https://github.com/likyoo/open-cd) projects.

## Citation

If you use the code or performance benchmarks of this project in your research, please refer to the following bibtex to cite TTP.

```
xxx
```

## License

The repository is licensed under the [Apache 2.0 license](LICENSE).

## Contact Us

If you have other questions❓, please contact us in time 👬

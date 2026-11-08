# 环境与运行路径

参考 CUDA 环境使用 PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128 和 RTX 5060 8GB。Windows CPU 依赖见 requirements-cpu-windows-py312.lock.txt，安装步骤见 [复现指南](README_zh.md)。

## 安装与路径

推荐Python 3.12。代码使用hashlib.file_digest，不支持直接使用Python 3.10及更早版本。GPU构建取决于系统与驱动，见[官方安装说明](https://pytorch.org/get-started/locally/)和[2.11历史版本](https://pytorch.org/get-started/previous-versions/)。不要复制作者整个Anaconda目录。

公开入口接受资源和输出路径。源码可迁移，数据和权重可分盘。ISW_CONFIG_CACHE可指定外部可写配置缓存，顶层复算入口自动将它设为输出目录。否则公开训练/评价在源码generated_configs生成小型配置。

部分历史配置和来源说明含作者盘符，可迁移入口会替换实际运行路径。resources.example.json仅给出本机可选默认位置，不是外部下载地址。

## 设备与精度

CPU 可用于推理和结果分析。浮点计算及近似并列查询的顺序可能随计算后端变化。论文的计算成本协议为指定 GPU 上的单图 FP32 推理，其他设备应报告各自的测量值。

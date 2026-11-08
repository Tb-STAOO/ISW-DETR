# ISW-DETR 复现指南

本仓库提供论文评价所需的代码、配置和运行说明。数据与参考权重可从 [GitHub Release](https://github.com/Tb-STAOO/ISW-DETR/releases/tag/sr-repro-20261108) 下载。复算输出写入指定目录。

## 安装

推荐Python 3.12。已在本机空白Windows虚拟环境安装CPU依赖并验证。参考GPU环境为PyTorch 2.11.0/cu128、RTX 5060。其他硬件的验证范围见PORTABILITY.md。

CPU环境：

```text
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements-cpu-windows-py312.lock.txt
```

GPU环境先安装与驱动匹配的PyTorch，再安装其余依赖：

```text
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-runtime.txt
python -m pip check
python tools/check_environment.py --device auto --output E:/ISW_Reproduction/environment.json
```

以下python指已安装依赖的解释器。路径可替换为其他盘或Linux绝对路径；输出使用新目录。

完整公开数据63组评价会保留全量预测，实测单套输出约23—26 GiB；建议输出盘至少预留40 GB。核心代码大小不代表完整运行的磁盘占用。内存较小时可将--bootstrap-workers设为1。

## 资源获取与导入

从 GitHub Release 下载参考评价资源。RESOURCE_ARCHIVES.json 记录资源包的 SHA-256 和用途。重新训练还需要向通讯作者取得对应的训练和初始化资源。

```text
python tools/import_resources.py --archives E:/SR论文/复现资源 --output E:/ISW_Reproduction/assets
```

导入时校验整包和逐文件指纹，拒绝不安全路径。可用--groups只导入所需分组：

| 资源组 | 用途 |
| --- | --- |
| private-inference | 自建数据三种子最终权重、评价划分和先验 |
| private-dataset | 自建数据原图、测量、训练及评价标注 |
| private-controls | BQR、置信度对照权重和MAP选择记录 |
| private-training | 找回的B0、三种子90轮权重、骨干初始化及衔接证据；重训时需要 |
| public-metadata | C-CCM原始元数据、固定成员索引和原README |
| public-checkpoints | C-CCM九组参考模型 |
| public-pretrained | C-CCM训练用官方COCO初始化 |

导入目录将连字符替换为下划线，例如assets/private_inference。公开原图不重复装入压缩包，可通过[原始数据记录](https://zenodo.org/records/4642577)按固定索引取得：

```text
python tools/prepare_cccm.py --metadata-root E:/ISW_Reproduction/assets/public_metadata --data-root E:/ISW_Reproduction/cccm --work-dir E:/ISW_Reproduction/cccm_preparation
python tools/run_cccm.py check --data-root E:/ISW_Reproduction/cccm
```

准备器核对源元数据指纹，保留8727张固定候选和2929/2829/2969划分，下载后检查成员CRC、大小、图像解码和哈希。已有PNG目录可用--image-source复用；--metadata-only只转换标注与先验，不能当成下载完成。

## 当前论文新增的配置重量诊断

public和all流程自动先运行仅重量诊断。若只核对42类配置赋值及95.39%类别正确率，导入public-metadata资源后运行：

    python tools/reproduce.py --phase public-diagnostic --assets-root E:/ISW_Reproduction/assets --output E:/ISW_Reproduction/mass_diagnostic

该模式只依赖Python标准库，不加载PyTorch、图像或模型。也可直接读取原元数据ZIP：

    python tools/run_mass_diagnostic.py --metadata-zip E:/SR论文/复现资源/public-metadata.zip --output-dir E:/ISW_Reproduction/mass_diagnostic_direct

两种入口均核对官方元数据指纹。输出为CONFIGURATION_MASS_TABLE.json/CSV及MASS_ONLY_CLASSIFICATION.json。固定测试划分为2,969帧，预期2,832帧分类正确（95.3856517%），42类对应40个参考重量值。该结果是重量输入的类别信息诊断，不能当作检测AP；其重量依据真实配置标注匹配，并非逐帧独立采集。

## 从参考权重复算论文结果

```text
python tools/reproduce.py --phase all --assets-root E:/ISW_Reproduction/assets --public-data-root E:/ISW_Reproduction/cccm --output E:/ISW_Reproduction/results --device cuda --measure-runtime
```

先运行配置重量分类诊断，再依次执行完整自建数据推理、117组对照、视觉/重量干预、共享候选池消融、10000次配对场景Bootstrap、类别/关联/采集时段分析、公开数据63组评价及2801张固定子集分析，最后生成数据图和可编辑LaTeX表。表格采用当前稿的三种子均值、最高/并列标量加粗、列居中、表外(a)/(b)标识和单行成本表头；表4为三个连续模型级条目。原始各种子结果及SD另存于CSV/TABLE_METRICS.json。源码指纹、日志及阶段结果保存在输出目录。类别、真实框和记录归属不进入模型前向。

还会从420个原始训练场景重建重量先验，校验28类分布重叠度，并检查10 g增量矛盾在六种评分路径上的完整查询回退。--measure-runtime同时单独重算batch-one验证精度。

只复算部分内容可选--phase private或public。CPU使用--device cpu并省略--measure-runtime；原计算成本协议需要CUDA，CPU运行不会填入历史速度值。测速时应保证GPU没有其他负载。

公开评价保留全部参考差异，不以新结果覆盖历史参考。同条件严格核对可单独用run_public_evaluation.py --reference-policy strict；新训练或新设备使用report保留差异。

## 重新训练

公开数据：共同基础20轮，标准续训和I–S辅助训练各10轮，三个种子均运行。

```text
python tools/run_cccm.py train-all --data-root E:/ISW_Reproduction/cccm --pretrained E:/ISW_Reproduction/assets/public_pretrained/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --device cuda --output E:/ISW_Reproduction/cccm_retraining
python tools/run_public_evaluation.py --data-root E:/ISW_Reproduction/cccm --weights-root E:/ISW_Reproduction/cccm_retraining --device cuda --output E:/ISW_Reproduction/cccm_retrained_evaluation --reference-policy report
```

自建数据：共享B0训练120轮，三种子视觉续训90轮，再进行I–S辅助训练，保留原验证选择与早停设置。

```text
python tools/train_scalevision.py --stage all --dataset-root E:/ISW_Reproduction/assets/private_dataset --backbone-pretrained E:/ISW_Reproduction/assets/private_training/pretrained/ResNet18_vd_pretrained_from_paddle.pth --device cuda --output E:/ISW_Reproduction/private_retraining
python tools/export_trained_weights.py --training-root E:/ISW_Reproduction/private_retraining --output E:/ISW_Reproduction/private_retrained_weights
python tools/reproduce.py --phase private --assets-root E:/ISW_Reproduction/assets --private-weights-root E:/ISW_Reproduction/private_retrained_weights --fit-private-controls --output E:/ISW_Reproduction/private_retrained_results --device cuda --measure-runtime
```

可选--stage b0/warm/is独立运行；后两者用--initializer明确给出前一阶段best.pth。初始化文件含EMA时加载EMA。找回的执行日志确认B0到90轮的加载路径；三种子的90轮EMA与最终模型各537个共有基础张量完全一致，因此无需重训不参与最终方法的历史试验头。原文件及指纹在private-training资源中。当前锁定数据不能替代尚未找到的训练当日数据快照，不宣称重建全部历史搜索。

骨干和其余参数的实际初始学习率均为10^-4，与当前论文一致。

--fit-private-controls会针对新检测器重新导出420个训练来源的查询，并训练BQR、置信度对照及选择MAP参数，然后完成全量对照、消融和统计。原参考权重不被覆盖。可用--external-public-results指定已经完成的63组公开数据JSON，一并生成公开数据表；重新评价公开模型时可用--public-weights-root指定新训练目录。

BQR/置信度对照及MAP：

```text
python tools/export_router_cache.py --dataset-root E:/ISW_Reproduction/assets/private_dataset --protocol-root E:/ISW_Reproduction/assets/private_inference/protocol --weights-root E:/ISW_Reproduction/assets/private_inference/weights --validation-inference-root E:/ISW_Reproduction/results/private --output E:/ISW_Reproduction/router_cache --device cuda
python tools/train_router_controls.py --method bqr --cache-root E:/ISW_Reproduction/router_cache --protocol-root E:/ISW_Reproduction/assets/private_inference/protocol --output E:/ISW_Reproduction/bqr_retraining
python tools/train_router_controls.py --method confidence --cache-root E:/ISW_Reproduction/router_cache --protocol-root E:/ISW_Reproduction/assets/private_inference/protocol --output E:/ISW_Reproduction/confidence_retraining
python tools/train_router_controls.py --method map --cache-root E:/ISW_Reproduction/router_cache --protocol-root E:/ISW_Reproduction/assets/private_inference/protocol --output E:/ISW_Reproduction/map_selection
```

对照训练采用420个来源的干净查询与原logit扰动，验证集选择，30轮、批量64。最终状态对比规则没有新增可训练参数。--smoke或--smoke-limit只用于入口诊断，其结果有明确标记，不能替代完整训练。

## 单张推理

```text
python inference/infer.py --image E:/images/example.jpg --transaction E:/measurements/example.json --checkpoint E:/ISW_Reproduction/assets/private_inference/weights/detector_seed42.pth --seed 42 --device cpu --output E:/ISW_Reproduction/example.json
```

缺失重量时省略--transaction。事务JSON仅允许七列原始测量、事件顺序、有效掩码及场景总重量等字段；未知监督字段被拒绝。批量格式和所有参数见各入口--help。

inference保留28类原生推理；runtime支持原训练及公开七状态接口；reference_sources保存核对过的原算法函数；tools提供可迁移调用和结果生成。原实现来源及函数一致性见reference_sources/SOURCE_PROVENANCE.json。历史YAML中的盘符由入口转换，请勿绕过入口直接运行历史配置。

通用COCO输出保留原始缺失指标哨兵：某尺寸区间没有真实目标时，COCO为-1，百分比JSON中相应为-100；它不代表负的有效AP。论文表格仅使用有定义的AP、AP50、AP75和AR100。


## 实验协议与结果输出

当前协议位于docs/PUBLIC_EVALUATION_PROTOCOL.md、docs/METHODS_STATISTICS_PROTOCOL.md及docs/SCALEVISION_DATASET_PROTOCOL.md。训练、模型和评分函数沿用原参考实现，仍使用既有七组数据/权重资源。

render_tables.py从传入的实际结果计算均值和加粗，不填入历史AP或耗时。cccm_table_main.tex采用当前稿的三行展示；原检测器的54.49/58.01等较强对照仍保留在完整结果和TABLE_METRICS.json。计时表使用新传入的benchmark，数值可随设备/运行变化。

复现入口生成计算结果与数据图。SOURCE_MANIFEST.json 记录当前源码指纹。

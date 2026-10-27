# VidVRD_stream1：片段完成后输出的 Stage 1 流式基线

仅实现 **VidVRD、GT 轨迹、PredCls**：已有片段特征 → 原空间编码器 →
固定历史窗口 → 因果时间编码器 → 当前片段多标签关系 → 在线关联。
不实现 Stage 2、长期记忆、token 压缩、KV cache、检测器、视觉主干、光流或逐帧边界回归。

## 1. 本地源码核对与兼容性

原项目 `../VidVRD_predication` 保持不变；本目录最初为空。
已核对原训练入口、两个阶段模型、数据集、特征生成、文本模板、参数解析、
后处理、评估器和启动脚本。原实现的主要事实：

- `train_stage1.py` 调用 `model_zoo/stage1.py`，只训练 `featEmbedding`。
- SpatialDecoder 在每个片段的 4 个区域 token 间交互；TemporalDecoder 在片段轴交互，
  原来没有 causal/padding mask。原物体表示是整个序列均值。
- 位置表长度 40；原物体/关系文本向量是固定向量，旧 checkpoint 不保存这些非持久 buffer。
- `mot_feat` 是首/中/尾框几何编码之差，不是光流。
- 旧 PredCls 使用缓存给定物体类别和 `sbj_scr/obj_scr`，不是模型物体分类输出。
- 原关联向后搜索完整序列，新实现不调用它。仅复制原评估器，移除机器专属默认路径，
  并处理空预测的 recall 边界，正常非空情况不改指标定义。
- 已核实生成器命名为 `<完整video_id>_<6位pair_id>.pkl`，用正则解析后缀，
  不截断 video_id。测试缓存没有真实跨 pair track ID，因此用稳定有序 pair_id，
  不按类别推断身份，也不跨 pair 合并。
- 原生成器保留 >=10 帧短尾，舍弃 <10 帧尾部；测试缓存 duration 已裁掉舍弃部分。
  不存在的尾部不能恢复，本项目不补造特征或训练绝对时间戳。

空间模块和时间模块保留原参数名及默认 768 维形状，包括原来未参与 forward 的
`spatial_decoder.relEmb`。该遗留参数没有梯度是预期行为，不应用假梯度填补。

## 2. 资源、依赖与配置

依赖当前环境的 Python 3.10+、PyTorch、NumPy；评估器还使用 tqdm。
不自动安装、升级或联网下载。无需加载旧特征提取器、InternVideo、光流等模型。

推荐结构（这些外部资源只读复用，不复制）：

```text
$DATASET_ROOT/
  feature/train_gt_30/*.pkl
  feature/test_gt_30/*.pkl
  data/openvoc_pred_class_spilt_info.json
  data/openvoc_obj_class_spilt_info.json
  data/test_relation_gt.json
```

也可分别指定 `--feature_root`（包含 train_gt_30/test_gt_30 的父目录）、
`--annotation`（原格式 GT 关系 JSON）、`--pred_class_split`、`--obj_class_split`。
`--dataset_root` 仅提供这些路径的默认值。`--prior` 被显式拒绝：本基线不做 prior 融合。
`--text_model_weights` 指本地 CLIP 文本模型所属 checkpoint，而不是特征提取视觉模型。
已有视觉特征不需要视觉模型路径，不会运行特征预处理，也不会清空特征目录。

所有新输出通过 `--output_dir` 指定，**必须在本项目内**（会检查 resolve 后的路径）。
同名运行会重写同名输出，请为不同实验使用不同目录。
配置包括 clip_len/window_size/batch_size/learning_rate/num_epochs/src_split/tgt_split、
clip_top_n、device、seed、损失权重等，详见 `python train_stage1.py --help`。

### 固定文本向量（优先推荐，无需加载任何预训练模型）

`--text_vectors fixed_text.pt` 使用 `torch.save` 保存的字典：

```python
{
    'subject': subject_vectors,    # [P,768]
    'object': object_vectors,      # [P,768]
    'predicate': predicate_vectors,# [P,768]
    'objects': object_class_vectors# [O,768]
}
```

行顺序严格对应 class-split JSON 的 id2cls 数字 ID。载入后逐行归一化。
真实实验必须使用与原 Stage 1 对应的固定 CLIP 文本表示，不能用随机向量代替。

可替代为 `--text_model_weights /path/to/ViT-L-14-336px.pt`。
此路径必须是已有本地文件，使用原项目的只读 CLIP 实现和原 prompt：
subject/object 角色模板、`An image of the visual relation {name}.`、
`An image of {object}.`。仅在构建固定文本时临时载入 CLIP checkpoint，
不提取视觉特征；构建结束后不在训练模型中保留 CLIP。
该可选路径依赖原项目 CLIP 的现有依赖及 tokenizer 文件，没有安装/下载兜底。

### 旧 Stage 1 checkpoint

在训练命令追加 `--init_checkpoint /path/to/original_stage1.pth`，同时提供文本向量
或本地 CLIP 权重。载入打印完整 loaded/missing/extra/shape_incompatible 名单。
空间或时间模块有缺失/形状冲突默认报错，显式 `--allow_partial_init` 才允许部分初始化。
不会通过模糊匹配偷偷跳过关键模块，Stage 2 的 pre_classifier/prompt 参数会被拒绝。
已识别的 Stage 2 checkpoint 不允许混用；无法为没有阶段元数据的任意外部文件推断来源。

新 checkpoint 包含模型、固定文本、窗口长度、clip_len、类别定义、配置、epoch 和
optimizer 状态。新 checkpoint 推理不再需要额外文本文件；窗口/片段/类别定义不符报错。
当前 `init_checkpoint` 是参数初始化，不是完整训练断点恢复（不会恢复 optimizer/scheduler）。

## 3. 数据形状、训练窗口与模型

每个缓存文件是 `[feature_dict, labels_or_pair_data]`，只读取：

| 特征 | 单个轨迹对形状 |
| --- | --- |
| clip_feat | [T,4,768] |
| bbox_feat | [T,4,24] |
| rel_feat | [T,42] |
| mot_feat | [T,42] |

T 是片段数量，不是帧数。默认 `clip_len=30` 必须匹配已有缓存划分，修改它不会重新切片。
`window_size=4` 包含当前片段；第 t 个样本取 `[max(0,t-K+1),t]`，只监督 t。
数据集只保存 `(文件索引, 当前位置)` 映射，最多 LRU 缓存 8 个 pair 的数组；
切片是视图，不把重叠窗口全部复制到内存。shuffle 仅打乱样本，不改变窗口内顺序。
保留无关系负样本；训练缓存没有 duration，不构造假绝对时间。

batch 右侧 padding，返回真实 seq_len，选取 `seq_len-1`。
空间输出关系 [B,T,4,768] 及主体/客体 [B,T,768]；物体分类表示取有效历史窗口的均值，
不包含未来或 padding。PredCls 最终类别仍取缓存给定类别，物体分类仅是辅助训练损失。

时间层对 `[B,T,4,D]` 按 `[B*4,T,D]` 建模，padding mask 使用 repeat_interleave(4)
对应同一样本的 4 个角色；上三角 bool mask 禁止访问未来。
窗口内位置始终 0..K-1，滑动后重新编号，K 必须在 1..40。
关系保持 `[subject,object,predicate,predicate]/2` 拼接、temperature=0.01、
多标签 sigmoid × sigmoid(interactiveness)，支持 base/novel/all，输出映射回全类别 ID。
不使用互斥关系 softmax。

损失 `L_rel + lambda_int*L_int + lambda_obj*L_obj`：
沿用原 gamma=2、未使用 alpha 权重的 focal 定义。
**原代码先对每个样本的有效片段/类别求 mean，再按样本求 sum**；现在每样本仅当前片段，
仍按样本求 sum，保持原批次尺度，不再额外除 batch_size。
物体损失仍是主体 CE mean + 客体 CE mean；各分支全 ignore 时返回有计算图的零，防止 NaN。
只优化 requires_grad 参数，不使用 `loss.requires_grad_(True)`。

## 4. 运行命令

### 与原 Stage 1 对齐的本机默认值

直接执行 `bash train.sh`，训练完成后执行 `bash infer.sh`，无需再填占位环境变量。
`local_defaults.sh` 集中设置可覆盖的本机默认值：

- DATASET_ROOT：按当前目录布局定位到 `/data/cyl/dataset/vidvrd`。
- 默认使用 `$HOME/.cache/clip/ViT-L-14-336px.pt` 生成固定文本，不假设存在 fixed_text.pt；
  若设置 TEXT_VECTORS 则优先使用该文件。已确认本机 CLIP 权重存在，不会下载。
- CUDA_VISIBLE_DEVICES=2，DEVICE=cuda:0，与原 train.sh 使用物理 GPU 2 一致。
  此处 cuda:0 是屏蔽后的逻辑编号；要使用物理 GPU 0，请设置 CUDA_VISIBLE_DEVICES=0。
- batch_size=32、learning_rate=1e-4、num_epochs=40、obj_loss_weight=0.2、
  int_loss_weight=0.1、src_split=base、tgt_split=all、seed=3407、clip_len=30，
  对齐原 `train_stage1.py` 的实际覆盖值和 parser 默认值，而非 Stage 2 启动脚本。
- 推理默认 CHECKPOINT 为本项目 `output/train/epoch_40.pth`，需先完成训练；
  不会误用原离线项目的 checkpoint。窗口仍为流式专属 window_size=4。

环境变量和末尾命令行参数可覆盖默认值。例如：

```bash
bash train.sh
bash infer.sh
CUDA_VISIBLE_DEVICES=0 bash train.sh --num_epochs 1 --limit_pairs 2 --max_steps 1
CHECKPOINT="$PWD/output/train/epoch_1.pth" bash infer.sh --limit_pairs 2
```

注意：本机默认路径只适配当前目录布局；搬迁项目后请设置 DATASET_ROOT。
`bash train.sh` 现在默认是 40 轮正式训练；本次修改未启动训练。

在本目录执行，路径由用户环境提供：

```bash
export DATASET_ROOT=/path/to/vidvrd
export TEXT_VECTORS=/path/to/fixed_text.pt
export DEVICE=cuda:0              # 无 GPU 时用 cpu
bash train.sh --num_epochs 25

# 小规模训练 + 同一流式逻辑验证（不会自动运行长训练）
bash train.sh --limit_pairs 2 --max_steps 1 --num_epochs 1 --batch_size 2

# 新 checkpoint 自带固定文本，窗口必须与训练一致
export CHECKPOINT="$PWD/output/train/epoch_1.pth"
bash infer.sh

# 完全显式的训练示例，可另加 --init_checkpoint 复用旧 Stage 1 参数
python train_stage1.py --dataset_root "$DATASET_ROOT" --text_vectors "$TEXT_VECTORS" \
  --output_dir output/experiment1 --clip_len 30 --window_size 4 \
  --batch_size 8 --learning_rate 1e-4 --num_epochs 1 --src_split base --tgt_split all \
  --clip_top_n 20 --device cpu --seed 3407
```

训练每轮保存 checkpoint，再调用和独立推理完全相同的 `runtime.replay`、
`StreamPredictor` 和在线关联。存在 annotation 时调用原 open-vocabulary 评估接口。
`--limit_pairs` 对训练和测试分别取排序后的前若干缓存文件；它只是 smoke 工具，
部分 pair 对完整 GT 的指标不能作为正式实验结果。
未提供/不存在 annotation 时仍导出结果，但不报告精度。

## 5. 在线语义、接口与关联

`stream.py:StreamPredictor.stream_step(video_id,pair_id,features,metadata,emitted_at_fid=None)`
是独立接口。features 是当前片段的无 B/T 维 Tensor 字典；metadata 只含当前可观测：
begin_fid/end_fid、sbj_cls/obj_cls、sbj_scr/obj_scr、该片段的 sbj_traj/obj_traj。
返回当前片段 scores 和本次 start/update/end 事件。

全部区间是半开 `[begin_fid,end_fid)`，输入必须在 end_fid 后释放。
离线缓存可提前读入内存；适配器只传当前切片，不把完整 duration、未来框、
未来类别或未来分数传给模型/状态机。按视频、end_fid、pair_id 稳定回放。
缓存只保存最多 K 个**未加位置编码、未时间编码的空间关系和物体表示**；
每次只空间编码一次新片段，时间编码器重算短窗口。

发生时间断档：在当前片段完成时关闭旧关系，清空该 pair 上下文，再处理新片段。
重叠/倒序输入被拒绝。不同视频、不同 pair 状态隔离；`reset(video_id)` 清除指定视频，
`reset()` 清除全部，包括已结束结果。不会根据完整 duration 提前关闭 pair。
缓存回放在该视频输入流 **EOF** flush，emitted_at_fid 使用最后已完成片段的时钟；
EOF 是明确的流结束信号，不宣称从缓存知道原始视频真实结束帧。
真实接入应在可观测视频结束/轨迹结束信号调用 `finish(video_id,clock,pair_id=None)`。
缓存没有真实轨迹结束事件，因此不提前推断；本版本不提供自动 timeout。

每个 `(video,pair,predicate)` 独立维护状态，多关系可同时激活：

- 与原项目一致，每个轨迹对、每个已完成片段按关系分数排序，取 top-20。
  用 `--clip_top_n` 调整；候选少于 N 时保留全部。不设置最低分数阈值。
- 未激活且进入当前 top-N：开启；已激活且仍在 top-N：延续。
- 已激活但退出当前 top-N：关闭，不把退出候选集的当前片段并入旧关系。
- 排名只使用当前完成片段的分数；只在目标 split 的有效 predicate 中排序，
  不让 split 外零分占位类别进入候选。排序采用与原项目相同的 NumPy argsort。
  不引入 gap tolerance，也不恢复离线未来遍历。
- 关闭后再出现是新实例；flush 关闭所有剩余关系。
- predicate_score 是已纳入片段关系分数的算术均值（短尾与完整片段等权）。
- triplet score = 开启时给定 sbj_scr × obj_scr × predicate_score，保持原 PredCls 约定。
  GT 缓存物体分数固定；不会调用模型物体分类改变最终类别或分数。
- 只累积已完成片段的框。已返回事件是独立快照，不被后续 update 回写。

输出文件：

- `*.segments.jsonl`：每个最新完成片段的固定分数和 emitted_at_fid。
- `*.events.jsonl`：start/update/end、pair/predicate ID、当前范围、累计分数和发出时钟。
- `*.json`：`{video_id:[{triplet,score,duration,sub_traj,obj_traj},...]}`，兼容原评估器。
- `*.metrics.json`：有 GT 时的 mAP/recall。训练另保存 config/checkpoint。

全视频 top-K 只在 EOF 后对最终结果排序，绝不参与在线开启/维持判断。
关闭事件 emitted_at_fid 允许晚于关系 end_fid。在线事件不是最终排序列表。

本版本已移除旧 `--on_threshold/--off_threshold` 参数；旧命令需改成 `--clip_top_n 20`。
不改模型结构、训练损失或 checkpoint 参数，可直接用已有 checkpoint 重新推理，
无需仅为更换后处理而重新训练。旧 checkpoint config 中记录的阈值不再参与推理。
重新评估建议指定新的 output_dir，保留原阈值策略的结果用于对照。
训练中的验证与独立推理均使用本 top-N 逻辑；已经运行中的 Python 训练进程需重启
才能采用更新后的代码，修改文件不会动态替换其已加载模块。

## 6. 测试与已验证范围

```bash
python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/smoke_cli.py
```

单测不联网、不依赖真实模型/数据：实际时间 Transformer 的因果性、padding 隔离、
有效物体均值、变长 batch 的当前位置、窗口滑动、逐步缓存与离线窗口一致性（含滑动）、
空间调用次数、状态隔离/reset、断档、开启/延续/关闭/再开启/多关系/flush、
事件时钟、有限 loss/梯度、负样本窗口索引、真实 filename 规则、短尾、评估轨迹格式、
checkpoint 报告与 Stage 2 拒绝、prefix consistency（不包含人为前缀 EOF flush）。
另测试非法 metadata 不污染缓存、top-N 不包含 split 外类别、低分仍入选、
高分但排名落选会关闭、空预测评估和原损失 reduction。
存在同级原项目时，还直接对照其 process_pred/association/format_：连续片段的最终
关系范围、轨迹及分数一致；无同级原项目时仅跳过此可选对照测试。

CLI smoke 在本项目临时目录生成 768 维合成缓存和随机测试文本，执行 **一个训练 step**、
自动流式验证、保存/重载 checkpoint、独立推理和 JSON 导出，再清理自己创建的临时目录。
随机文本只用于程序验证，不能作为真实精度实验。测试日志位于 `output/verification/`。
另对本地一个真实测试 pair 核实了缓存形状和 metadata；旧 Stage 1 checkpoint 的
空间/时间参数兼容性检查结果见该目录日志。

未运行完整真实训练、完整数据评估、GPU 性能测试；本地 CLIP 文本构建分支尚未实跑。
本阶段是 **GT 轨迹条件下的缓存特征实验**，不是在线检测跟踪系统。
每个片段的几何特征包含末帧信息，因此必须等片段结束，不能声称逐帧即时预测。
边界精度仍是原片段级，不解决逐帧精确定位。
缓存推理不包含解码、检测/跟踪或视觉特征提取时间，不能代表端到端实时性能。

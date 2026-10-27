# VidVRD_stream1：片段完成后输出的 Stage 1 流式基线

### Novel 评估口径修正

`runtime.replay` 的评估器现使用当前 `model.tgt_split`，与实际预测候选范围一致。
双 split 验证依次预测 novel/all，GT 也分别按 novel/all 过滤，而不是始终使用启动
参数中的 tgt_split=all。旧版本日志中 Novel 指标实际上使用 all GT，数值被低估，
对应 Mean mAP 和最佳 checkpoint 选择也受影响；不要与修正后的结果直接比较。
模型权重、训练损失、top-N 和在线关联没有改变，已有 checkpoint 可直接重新评估：

```bash
CHECKPOINT=/path/to/existing/checkpoint.pth bash infer.sh --output_dir output/reeval_correct_split
```

该命令记录正确口径的 novel/all 和平均 mAP，不重训、不改写旧权重或旧日志。
重新训练请使用新的 run 目录，避免把旧口径的 best mean_map 与新口径混用。
修正后可重评已有最佳权重，但不能保证它仍是整个历史训练过程的真正最佳轮次，
因为旧版本只保留了错误口径下的一个最佳 checkpoint。

### 训练输出目录更新

`train.sh` 每次启动默认使用 `output/run_日期_时间_进程号/`，内部仍只有 log 和
checkpoint 输出目录，避免新配置与已有 checkpoint 的配置保护检查冲突。
启动日志 Configuration 会显示实际 output_dir。旧文件不删除、不覆盖。
可用 `OUTPUT_DIR=output/my_experiment bash train.sh` 或末尾 `--output_dir` 显式指定。
此时若目录内已有不兼容配置的 checkpoint，仍会报错，以保护已有实验。
推理不会自动选择新 run，需显式设置该次训练的 CHECKPOINT，例如：

```bash
CUDA_VISIBLE_DEVICES=6 bash train.sh
CHECKPOINT="$PWD/output/run_实际日期_时间_进程号/checkpoint/rela_vidvrd_bs32_lr0.0001_dim512_None_rel_mask_lan_v2d_mot_v3d_clip_intern_bbox__stage1.pth" bash infer.sh
```

下文旧的固定 output/checkpoint 示例适用于之前的运行或显式指定 OUTPUT_DIR=output。

## 缓存推理批处理

推理及训练中的验证回放现支持视频内/跨视频 batch；训练采样、损失、模型结构、
参数名、checkpoint 和原在线 top-N 关联算法不变。新增参数独立于训练 batch_size：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| inference_batch_size | 32 | 每次前向最多处理多少个当前轨迹对片段 |
| max_active_videos | 1 | 同时载入并调度的视频数；设为 1 只做视频内 batch |
| max_videos | 0 | 按视频 ID 排序取前多少个视频评估，0 表示全部 |

```bash
# 串行对照（同样使用新的批量接口，独立旧参考见 tests/reference_serial.py）
bash infer.sh --inference_batch_size 1 --max_active_videos 1 --max_videos 2
# 同一视频同一 end_fid 的多个 pair 合并前向
bash infer.sh --inference_batch_size 32 --max_active_videos 1
# 多路视频填充 batch，每路独立时钟，不要求帧号对齐
bash infer.sh --inference_batch_size 32 --max_active_videos 4
```

`stream_step_batch(events)` 接收事件字典列表，每项键与单事件接口一致：
video_id、pair_id、features、metadata、可选 emitted_at_fid；返回等长且同顺序的结果列表。
`stream_step(...)` 通过一个元素的 batch 复用它。接口使用 eval + inference_mode，
一个 batch 中同一 (video,pair) 最多一次，且每视频只能出现一个释放时刻组；
非法重复 pair/多时刻输入在更新缓存之前报错。外部调用者负责释放顺序，空 batch 返回空列表。

调度器稳定按 (end_fid,pair_id) 推进每个视频；一次只取该视频当前时间组，组太大可分批，
未处理完不推进该视频。不同视频按稳定轮转顺序调度，前一 batch 处理完才产生下一批。
每个 pair 的空间缓存、原始物体表示和关联状态均独立；历史空间特征不重编码，
空间输出按样本 clone 后缓存，避免一个残留 pair 引用整个 batch 的 GPU 存储。
变长窗口右 padding，真实 seq_lens 传给现有模型，仍取 seq_lens-1、有效物体均值，
滑动窗口重新编号。完成 batch 后按输入顺序更新关联，EOF 在最后一批关联完才 flush；
断档规则不变，不增加超时/轨迹终止猜测，不为凑 batch 提前关闭或等待。

每批一次将完整分数矩阵传到 CPU，随后复用原 NumPy argsort 和有效类别范围；
不改为 torch.topk，以保留现有并列排序规则。没有稀疏丢弃活跃关系分数，
候选、事件与最终 `{video:[{triplet,score,duration,sub_traj,obj_traj}]}` 结构不变。
默认入口仍遵守只写 log/最佳 checkpoint 的政策，不新增预测文件；事件从接口返回，
最终 JSON 兼容字典在内存中用于评估。

加载先建文件路径索引，不读全部特征；最多持有 max_active_videos 个完整视频的
特征/当前和未来事件列表。模型只接收已释放片段，未来数据不参与模型和关联决策。
视频 EOF 后释放其特征、GPU 工作缓存和关联工作状态，再载入下一路。
完整视频输入仍可较大，且全体视频最终评估结果会累积在 CPU，GT JSON 也完整载入，
因此不宣称整个程序严格常数内存。
`max_videos` 指定时 GT 与预测都限制到选定视频；`limit_pairs` 仍是部分 pair smoke
选项，不会相应删减该视频 GT，其指标不可当作完整正式结果。

跨视频 batch 主要优化多路吞吐量，不等于降低单路响应延迟；缓存特征回放
不等于包含解码、检测/跟踪及视觉特征提取的真实端到端在线检测。
未引入混合精度、计时、profiling 或吞吐量统计。

合成测试保留改造前独立串行实现，比较原始当前分数、缓存、关联事件、最终轨迹，
覆盖窗口滑动、不同长度、断档、prefix、EOF/reset 隔离、空输入、不满 batch、
惰性加载上限、重复 pair 拒绝与完全并列的文本类别分数。
可运行真实子集正确性对照（不统计性能）：

```bash
OMP_NUM_THREADS=1 python tests/compare_cached_batch.py \
  --dataset_root "$DATASET_ROOT" --init_checkpoint "$CHECKPOINT" \
  --max_videos 2 --inference_batch_size 8 --max_active_videos 2 --device cpu
```

该脚本分别对 novel/all 比较冻结串行参考、batch=1、视频内 batch、跨视频 batch
的逐片段分数、事件、最终关系和 mAP/Recall。批量 GEMM 的正常浮点差异允许约 2e-5，
候选 ID、事件范围与轨迹不得变化；不通过放宽候选一致性来掩盖近似并列差异。

本次实际验证：24 项合成单测、768 维 CLI smoke 通过；本地真实 streaming checkpoint
和排序前两个完整视频（未使用 limit_pairs）在 CPU 上对 novel/all 分别完成冻结串行、
batch=1/active=1、batch=8/active=1、batch=8/active=2 对照。
novel mAP=0.0769230798、Recall@50/100=1.0；all mAP=0.2486324907、
Recall@50/100=0.6666667，三种执行配置均一致。这仅是小子集正确性检查，不是全数据结果。
当前 CUDA 不可用，未验证 GPU 批处理、全测试集一致性或实际性能提升。

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
- 推理默认 CHECKPOINT 为本项目 `output/checkpoint/rela_vidvrd_bs32_lr0.0001_dim512_None_rel_mask_lan_v2d_mot_v3d_clip_intern_bbox__stage1.pth`，需先完成训练；
  不会误用原离线项目的 checkpoint。窗口仍为流式专属 window_size=4。

环境变量和末尾命令行参数可覆盖默认值。例如：

```bash
bash train.sh
bash infer.sh
CUDA_VISIBLE_DEVICES=0 bash train.sh --num_epochs 1 --limit_pairs 2 --max_steps 1
bash infer.sh --limit_pairs 2
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
export CHECKPOINT="$PWD/output/checkpoint/rela_vidvrd_bs32_lr0.0001_dim512_None_rel_mask_lan_v2d_mot_v3d_clip_intern_bbox__stage1.pth"
bash infer.sh

# 完全显式的训练示例，可另加 --init_checkpoint 复用旧 Stage 1 参数
python train_stage1.py --dataset_root "$DATASET_ROOT" --text_vectors "$TEXT_VECTORS" \
  --output_dir output/experiment1 --clip_len 30 --window_size 4 \
  --batch_size 8 --learning_rate 1e-4 --num_epochs 1 --src_split base --tgt_split all \
  --clip_top_n 20 --device cpu --seed 3407
```

训练每轮调用和独立推理完全相同的 `runtime.replay`、`StreamPredictor` 和在线关联，
分别进行 novel/all 候选预测及评估，记录两组性能后，按 `(novel mAP + all mAP)/2`
选择最佳 checkpoint。原项目重复评估两次同一 GT 测试集的 novel/all；这里各评估一次，
避免重复计算，平均值定义等价。两次预测分别使用各自 split 的 top-N，而非同一结果后过滤。
`--limit_pairs` 对训练和测试分别取排序后的前若干缓存文件；它只是 smoke 工具，
部分 pair 对完整 GT 的指标不能作为正式实验结果。
训练和独立推理都要求有效 annotation；缺失即报错，不会产生无指标的最佳 checkpoint。
两者均输出 novel/all 指标，不受单独的 tgt_split 限制；底层 replay 仍支持单 split。

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

### 输出文件与进度

默认只在 output 下创建 log 和 checkpoint 两个目录：

```text
output/
  log/rela_vidvrd_bs32_lr0.0001_dim512_None_rel_mask_lan_v2d_mot_v3d_clip_intern_bbox__stage1.log
  log/rela_vidvrd_bs32_lr0.0001_dim512_None_rel_mask_lan_v2d_mot_v3d_clip_intern_bbox__stage1_infer.log
  checkpoint/rela_vidvrd_bs32_lr0.0001_dim512_None_rel_mask_lan_v2d_mot_v3d_clip_intern_bbox__stage1.pth
```

训练日志包含带时间戳的配置、epoch、平均 batch loss、novel/all 的 mAP、Recall@50/100、
两组 mAP 算术均值、历史最佳均值及保存路径；性能以百分数记录。
名称中的 dim512/None/特征串是与原项目一致的命名约定，不代表实际空间 token 是 512 维，
实际仍是 768 维。batch_size/learning_rate 会随配置改变文件名。
日志追加写入，进度条不会写入日志。仅当 mean mAP 严格超过历史最佳时覆盖同一个 pth，
首轮即使为零也会保存；相同分数不覆盖。checkpoint 内仍保存配置、文本、类别、窗口、
optimizer、epoch 和指标，不再额外写 config 文件。已有同名 checkpoint 配置不同则报错，
请指定另一个 output_dir，不会静默覆盖其他实验。

不再写逐 epoch checkpoint、预测 JSON、事件/片段 JSONL 或 metrics JSON。
结果与事件仅在内存供在线逻辑和评估使用，stream_step 返回接口不变。
旧版本遗留文件不会自动删除，如需清理请先自行备份。
训练进度按 batch，推理进度按已完成视频，均用 tqdm 显示总量和完成比例；
`--progress_width 100` 设置总宽度（至少 40 列）。

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
自动流式验证、保存/重载最佳 checkpoint、独立推理，并检查只生成 log/pth，
再清理自己创建的临时目录。另有严格改善/持平/下降情况下仅保留最佳 checkpoint 的测试。
随机文本只用于程序验证，不能作为真实精度实验。历史测试日志曾保存在 `output/verification/`，
新训练/推理不会再创建该目录。
另对本地一个真实测试 pair 核实了缓存形状和 metadata；旧 Stage 1 checkpoint 的
空间/时间参数兼容性检查结果见该目录日志。

未运行完整真实训练、完整数据评估、GPU 性能测试；本地 CLIP 文本构建分支尚未实跑。
本阶段是 **GT 轨迹条件下的缓存特征实验**，不是在线检测跟踪系统。
每个片段的几何特征包含末帧信息，因此必须等片段结束，不能声称逐帧即时预测。
边界精度仍是原片段级，不解决逐帧精确定位。
缓存推理不包含解码、检测/跟踪或视觉特征提取时间，不能代表端到端实时性能。

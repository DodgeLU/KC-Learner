# Legacy regression reference

`project_c/` is not part of this repository. Where a local checkout exists, `SDATT-PLUS/project_c/` is a read-only regression reference. KC-Learner does not modify it, copy its checkpoints, or import it.

## EdNet

正式 legacy regression target：当前 **dev5000** 路径。

- Split：`ednet_split_v1`，目录
  `project_c/data_proc/ednet_kt1/ednet_kt1_pilot_v1_from_v2_split_v1`
- Cohort：dev5000
- 模型：IRT、AR-KT、DKT-Q、DKT-QC、DKVMN-Q、DKVMN-QC
- 冻结汇总中的神经模型 seeds：42、43、44
- IRT / AR-KT checkpoint：
  `project_c/final_valid_pt/EdNet/IRT/ednet_irt_dev5000/`
  `project_c/final_valid_pt/EdNet/AR-KT/ednet_ar_kt_dev5000/`
- 结果汇总：`project_c/final_test_results/EdNet/`

Legacy preprocessing id：`ednet_kt1_preproc_v1`。该流水线把 `questions.csv` 的 `tags="-1"` 解析成 tag id `-1`，之后可能映射成一个稠密 KC 下标。这个行为保持不变。

## Corrected EdNet preprocessing

KC-Learner id：`ednet_question_tags_corrected_v1`。

```text
tags = "-1"        →  无 KC 标注  →  kc_ids = ()
blank / missing    →  kc_ids = ()
tags = "5;-1;2"    →  报错（mixed -1 仍然非法）
非整数 token       →  报错，信息包含 question_id、raw tags、invalid token
```

合法 tag 按源顺序保留，包括重复 token。`"5;2;5"` 仍是 `(5, 2, 5)`。不去重。公开重建身份是 `ednet_kt1_corrected_v1`：直接读每个 learner 的原始 KT1 CSV，保持文件行序，bundle 是 `(user_id, solving_id)`。它不读取 interim 分片。dev5000 是历史 `ednet_split_v1` TRAIN∪VALID 用户 ID 字典序的前 5000 人。用户 ID 列表只写在本地 `generated/`，仓库只保留其 SHA-256。

和 legacy parquet 的差异分成三类。有意修正：恢复原始行序、恢复 `(user_id, solving_id)` bundle、把 `tags="-1"` 当成缺失 KC。沿用的资格规则：空答案或非法答案仍然过滤，222 行空答案不是新的剔除规则。表示差异：canonical 列，以及 parquet 里保存源 ID；发表用的 dense index 另由 `ednet_split_v1` TRAIN+VALID 的 12,259 个题目和去掉 `-1` 后的 188 个 KC 决定。正式实验协议见 `docs/final_experiment_protocol.md`。

> KC-Learner uses source-faithful preprocessing for EdNet. Valid dataset-provided annotations and source interaction order are preserved. Dataset-specific missing/sentinel encodings are normalized, while potential source-data irregularities are reported rather than silently corrected.

- 重复作答允许存在。同一 `question_id` 的多次观测不会被自动当成重复记录删掉。
- 题目元数据里的重复 KC token 原样保留。
- `solving_id` 只定义 bundle `(learner_id, solving_id)`，不用来重排历史。
- 回放顺序是过滤后的源记录顺序。同一 bundle 内也保持该源顺序。
- timestamp 只作元数据和诊断。solving_id 逆序或 timestamp 逆序会记入 diagnostics，不会因此丢行或重排。
- `user_answer="-1"`、空答案、异常答案、缺失题目元数据会被过滤并计入原因，不写成 `correct=0`。

`questions.csv` 的 `bundle_id` 是内容标识，记为 `content_bundle_id`。它不是交互 bundle。

通用 `assign_fractional_bundle_split()` 只给合成数据或自定义数据写 split 标签，不改行序。它按 bundle 首次出现的顺序做比例切分。它不用于正式 EdNet legacy regression，也不重写 dev5000 parquet。

## Formal EdNet regression contract

Legacy regression reference 是这三项的组合，不再重新生成：

```text
formal dev5000 existing split labels
+ physical row order stored inside each split parquet
+ actual legacy bundle execution behavior
```

不要按 `solving_id` 或 timestamp 重排，也不要用 fractional splitter 代替已有 split 标签。

dev5000 的 TRAIN / VALID / TEST 里，许多 `(user_id, solving_id)` 在 learner 自己的物理子序列中并不连成一块。正式 runner 不会把这些行收成一个逻辑 bundle。执行单位是物理顺序上的连续片段：`(student_idx, solving_id)` 一变就结束当前片段。同一片段内先对所有行预测，再更新状态，然后进入下一个片段。

IRT 与 AR-KT 的 `EdNetARTKTModel.run_streaming` 直接走整个 DataFrame。其他 learner 的行插在中间时，也会切断当前片段。DKT 与 DKVMN 先按 learner 取出子序列，再用 `list_user_bundles` 按 solving_id 的连续变化切 `BundleSpan`，其他 learner 的插入不会切断。

## Corrected EdNet identity

公开的新预处理保持：`tags="-1"` 为缺失 KC，合法 tag 的顺序和重复次数保留，源记录顺序保留。它不负责逐数复现 legacy 预测。不要为了 parity 把公开 adapter 改回历史 `-1` 行为。

AR-KT 的预测级 parity 应先用没有 `tags="-1"` 交互的固定序列，或单独的 legacy-only fixture。

## ASSISTments 2017

正式 regression target：`ASSISTMENTS2017_MAIN_V1`。加载器是 `kclearner.data.adapters.assistments2017`。它不读 `strict.csv`，也不计算 TEST 指标。

- 源：`project_c/data_proc/assistments2017/primary.csv`（`assist17_primary_v1`，`attempt_count == 1`）
- 排序：按 `(student_id, timestamp)` 稳定排序
- Bundle：`(student_id, timestamp)`。模型用的 `group_id` 是 TRAIN+VALID 上这个键的稠密编号
- Split：每个 learner 名义 70/10/20（`B_temporal_v1`）。跨 TRAIN/VALID 的 bundle 整组移到 VALID（`assist17_b_temporal_bundle_safe_v1`）。VALID/TEST 边界不闭包
- 词表：TRAIN+VALID 的字符串排序。2909 个题目，102 个 KC，1709 个学生
- 行数：TRAIN 299,907，VALID 42,847，名义 TEST 85,741
- 题目 OOV 只出现在名义 TEST：949 行不进指标。IRT / AR-KT 在 eval 仍用 `problem_idx = 0` 走一遍并更新 theta（AR-KT 还更新该技能残差），`b` 冻结。DKT / DKVMN 对这些行是 state-transparent：不预测、不更新 LSTM 或 value memory
- AR-KT `eta_r = 0.20`，`use_time = False`。残差是每个技能在 bundle 内的 `(y - p_global)` 均值，不是 EdNet 的 token 公式
- 神经 checkpoint：`final_valid_pt/assistments2017/{DKT,DKVMN}/seed_42/{q,qc}/`

`CP2_AR_KT_V0.3`（`eta_r = 0.50`）是历史协议，不是这条回归路径。

## KC-Learner IRT / AR-KT port

`kclearner.models` 里的 streaming 是 `project_c/models/ednet_art_kt.py` 中 `EdNetARTKTModel.run_streaming` 的移植。它按输入的物理行序扫描，执行键是 `(student_idx, group_id)`。它不调用 `replay_bundle_pre_state()`。

IRT 是 `use_residual=False`、`eta_r=0`。AR-KT 默认是 EdNet token 残差：`use_residual=True`、`eta_r=0.20`、`use_time=False`。ASSISTments checkpoint 另选 `assistments_skill_mean`，不改 EdNet 默认。

公开 EdNet adapter 的 `tags="-1"` → `kc_ids=()` 不变。legacy parity 不改这条语义；本地窗口直接排除 raw tag `-1`。

本地对照见 `integration/FIXTURE_MANIFEST.md`。它只在显式运行 `python -m unittest discover -s integration -v` 时执行。

原生状态文件是 `kclearner_ednet_streaming_state_v1`。`load_legacy_checkpoint` 读取 EdNet 与 ASSISTments MAIN_V1 的 `.npz`。IRT 快照没有 residual 数组；AR-KT 快照包含 `r_post`、`r_seen`、`last_r_update_ts`。`model_id` 决定残差更新。这不改变公开 EdNet adapter。

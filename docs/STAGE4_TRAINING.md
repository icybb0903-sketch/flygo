# 阶段 4：离线线性读出训练与评估

## 结论边界

本阶段完成了一次可复现、可审计的离线教师蒸馏实验：固定 MaleCNS v1.0
结构图和本项目工程 LIF 动力学为每个 9×9 围棋局面产生 128 个活动特征，再用
这些特征拟合一个 `128 -> 82` 的线性读出。教师只是现有
`deterministic-capture-first-v1`，仅在离线数据生成阶段提供白方标签。

这不证明果蝇、MaleCNS 或该读出“学会”“理解”围棋，也不证明棋力强。连接组提供
真实结构约束；刺激编码、LIF 参数、输出池、训练标签和线性读出都是工程设计。

## 可复现流程

在项目根目录、项目虚拟环境中运行：

```powershell
.\.venv\Scripts\python.exe scripts\train_neural_go_readout.py `
  --games 8 --max-moves 72 --duration-ms 12 `
  --seed 20260921 --validation-fraction 0.25 --ridge 0.01
```

若缓存已存在且通过全部哈希校验，可只重复训练：

```powershell
.\.venv\Scripts\python.exe scripts\train_neural_go_readout.py `
  --reuse-cache --ridge 0.01
```

流程如下：

1. 从空棋盘生成 8 局合法游戏。黑方按固定种子从合法点中探索，并以固定概率优先
   捕获；白方按现有 capture-first 基线行动。
2. 只把白方行动前的状态作为监督样本。每个标签均由当时的合法动作产生。
3. 用 `SHA-256(seed, game_id)` 对整局排序，将完整游戏分到 train 或 validation，
   不允许同一局的状态跨集合。
4. 对每个状态调用同一个 `src.malecns_dynamics.simulate_go_encoding`，在真实
   `139,662` 节点、`5,536,347` 边 CSR 图上运行工程 LIF；本次窗口为 12 ms。
5. 直接读取该帧完整活动数组产生的 128 个固定输出池特征。显示子集不参与训练。
6. 对 one-hot 82 动作目标做闭式 ridge 最小二乘，`ridge=0.01`，截距不惩罚。
7. 通过 `src.malecns_policy.write_policy_checkpoint` 写出并用严格加载器校验检查点。

运行时检查点只有 `float32[82,128]` 权重、`float32[82]` 偏置及溯源清单；其
`runtime_guarantees` 明确为 `teacher_access=false`、`baseline_access=false`、
`fallback_controller=false`。运行时会对 checkpoint、frame、encoding、输出池和
合法 mask 严格校验，不会调用教师兜底。

## 缓存审计

`samples.jsonl` 每行包含：完整 `GoState`、整局 ID、split、ply、教师动作、合法
mask、捕获数、LIF seed、图和参数标识，以及以下逐样本哈希：

- `state_sha256`：完整状态（含局面历史）；
- `encoding_sha256`：围棋神经编码；
- `frame_sha256`：确定性 LIF 帧 ID；
- `label_sha256`：教师 ID 与动作；
- `features_sha256`：输出池定义与 128 特征；
- `record_sha256`：整条 JSONL 记录。

`feature_row` 把 JSONL 与 `features.npy`、`labels.npy` 严格对齐。加载缓存时先校验
dataset manifest，再校验三个文件哈希、数组形状、行序、记录哈希、状态哈希和标签
哈希。特征和标签均禁止 pickle。

本次数据集 ID：
`e6c814136ccdc4fe4eb9c5cc3192dce1e8f09073e2987bdaec43aa178bf3cf76`。

| 项目 | 数值 |
|---|---:|
| 样本总数 | 288 |
| 训练 / 验证样本 | 216 / 72 |
| 训练 / 验证整局 | 6 / 2 |
| 教师动作覆盖 | 63 / 82 |
| 教师捕获样本 | 28 |
| 后 1/4 局面样本 | 72 |
| 输出特征 | 128 |

## 指标定义与实测结果

- `top1_accuracy`：未施加合法 mask 的最高 logit 是否等于教师标签；
- `top5_accuracy`：教师标签是否在未施加 mask 的最高 5 个 logit 中；
- `legal_action_rate`：未施加 mask 的 top-1 是否是规则合法动作；
- `teacher_agreement_rate`：施加合法 mask 后，运行时会选择的动作是否等于教师。

| split | 样本 | top-1 | top-5 | 合法动作率 | mask 后教师一致率 |
|---|---:|---:|---:|---:|---:|
| train | 216 | 12.9630% | 41.6667% | 39.3519% | 42.1296% |
| validation | 72 | 6.9444% | 30.5556% | 16.6667% | 34.7222% |

首轮真实端到端运行耗时为 **213.194 秒**。随后为补充逐样本特征哈希与教师标签
合法率，使用相同模型和参数重新生成最终审计产物。最终 checkpoint manifest
SHA-256 为：
`4f6f62870ce7bc5db9bf199f6e17b21f3db1796edabb0fcb53f8ec5b3c225746`。

两轮的特征数组、标签数组和线性权重逐字节一致；变化来自最终 JSONL 增加的审计
字段及其传递到 dataset/checkpoint manifest 的哈希。训练 Agent 在第二轮落盘后因
Codex 用量限制中止，以下最终哈希由主 Agent 直接从磁盘重新计算，并重新执行严格
checkpoint 加载和 146 项测试确认。

验证指标很低，特别是未 mask top-1 合法率仅 16.67%。这应如实理解为：当前小样本、
短 LIF 窗口和线性读出只学到有限的教师相关信号；合法 mask 保证部署动作合法，但不
会把弱模型变成强棋手。

## 产物与 SHA-256

| 文件 | SHA-256 |
|---|---|
| `data/training/stage4/features.npy` | `572aa0eb8c4142fda9bffb3372a0927bf18ed59da0afc2b5c50f6a9ba45a39a2` |
| `data/training/stage4/labels.npy` | `f918056a0d2349d009d9919b3e1d9c7b0a3be1d50b2219bcd51050b18b02ed67` |
| `data/training/stage4/manifest.json` | `9fccfbf775cfcb5b6d93e8d6d8912d4dc098dc3bec88927290cbcd2527f7a2c6` |
| `data/training/stage4/samples.jsonl` | `bafdbcfd3e6653dc7feec1a3bceb16e0d4a02717e0f9a1cb345d9a04980d4e39` |
| `data/training/stage4/training_result.json` | `25c0495a0d600686047cdac45528868444f844068a997b2d7f44c2c1a04077e3` |
| `data/checkpoints/stage4/weights.npy` | `dd9161b871b2ab9540039da73539e5f74e0148c8bb68a91007b6f81b20f98138` |
| `data/checkpoints/stage4/bias.npy` | `63abbbe199f7d58c7085f55664ad03aa9ad876b60161a15bd1903cf92f9b7bc7` |
| `data/checkpoints/stage4/manifest.json` | `4f6f62870ce7bc5db9bf199f6e17b21f3db1796edabb0fcb53f8ec5b3c225746` |

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

实测：`Ran 146 tests ... OK`。阶段 4 专项测试使用小型合成缓存，覆盖整局切分的
确定性和不泄漏、岭回归字节级重复、四项指标、缓存篡改拒绝、checkpoint 严格加载，
以及权重文件篡改后的 fail-closed 拒绝。

## 已知限制

- 只有 8 局、288 个状态，虽然覆盖 63 个动作、捕获和后期局面，仍不足以代表 9×9
  围棋状态空间，尤其未覆盖 pass 标签和全部 82 个动作。
- 验证集只有 2 个完整游戏，指标方差会很大；没有多随机种子置信区间。
- 教师本身只是 capture-first 启发式；对它的拟合不是强棋力证据。
- 当前只训练外部线性读出，MaleCNS 图权重和 LIF 参数均被冻结；不能描述为果蝇神经
  系统发生了生物学习。
- 输出池和动力学参数是可复现的工程选择，未经生理数据校准。

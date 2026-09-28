# Stage 7 白方策略评测与冻结计划

## 结论边界

当前默认运行的 20 局只能标记为 **development evidence（开发证据）**。无论胜率或
`p` 值多好，都不得称为 Stage 7 正式通过、最终验收或已冻结模型。脚本默认输出
`evaluation_tier: development`、`status: development_completed`，避免把小样本开发集
误写成最终结果。

本阶段只评估网页真实角色：MaleCNS 策略执白，合法均匀随机策略执黑。因此结果不能
外推为黑白通用策略，也不能外推为对搜索型围棋程序的棋力。

## 同 seed 经验基线

每个神经策略对局都额外运行一局相同 seed 的 `random White vs random Black`：

- 黑方随机流使用与 MaleCNS 执白对局相同的 seed 绑定；
- 白方使用独立但由同一 game seed 确定的随机流；
- 基线不加载或调用神经动力学、检查点或策略推理；
- 报告同时给出随机白胜率和
  `MaleCNS 白胜率 - 随机白胜率` 的百分点提升。

这比只与理论 `0.5` 比较更贴合当前规则、贴目和固定步数裁决产生的白方经验优势。

## 正式冻结门槛（不可由 CLI 降低）

只有以 `--evaluation-tier final` 运行且以下条件全部满足，报告才可标记 `passed`：

1. 对局数 `n >= 100`；
2. MaleCNS 白方胜率 `>= 0.80`；
3. 胜率 95% Wilson 区间下限 `>= 0.70`；
4. 双方连续停着形成的自然终局比例 `>= 0.90`；
5. 相对同 seed 随机白方经验基线的胜率提升 `>= 0.15`（15 个百分点）；
6. 每局 seed 唯一，并且与所有指定参考开发/训练报告的 game seed 零重叠；
7. MaleCNS 每次决策前的 `board_hash_before` 与所有指定参考报告中的决策状态哈希
   零重叠；
8. 运行期继续满足未调用 teacher、baseline controller 和 fallback 的保证。

正式运行必须通过一个或多个 `--reference-report` 提供需要排除的既有开发/训练对局。
如果没有参考报告，重叠审计标记为未执行，正式冻结门槛必定失败。初始空棋盘不参与
比较；比较对象是 MaleCNS 实际决策前的局面，因此不会因所有棋局共有初始状态而产生
伪重叠。

旧的 `--minimum-games`、`--minimum-win-rate`、`--minimum-wilson-lower` 参数仅为保持
CLI 兼容并展示开发视图，不能降低上述冻结门槛。

## 统计解释

报告保留 `one_sided_binomial_p_value_vs_half`，但它只作描述。`p <= 0.05` 不是主门槛，
也不能替代样本量、Wilson 下限、自然终局、经验随机基线和数据隔离审计。

## 可复核哈希

报告包含两个哈希：

- `semantic_core_sha256`：排除耗时、输出路径和其他 `*_path` 本机字段，用于跨机器、
  跨运行比较相同证据；
- `report_sha256`：保留完整产物级记录，包括耗时等运行信息。

因此不能用耗时或磁盘路径变化冒充实验内容变化。

## 建议命令

开发运行（默认 20 局，不能作为最终验收）：

```powershell
python scripts/evaluate_stage7_white.py
```

正式冻结候选（昂贵，至少 100 个全新 seed）：

```powershell
python scripts/evaluate_stage7_white.py `
  --evaluation-tier final `
  --seed-count 100 `
  --base-seed <预先冻结的新起始 seed> `
  --reference-report data/evaluation/stage6/balanced_candidate_report.json `
  --reference-report data/evaluation/stage7/white-development.json `
  --output data/evaluation/stage7/white-final.json
```

正式运行前应先锁定 checkpoint hash、graph manifest hash、seed 清单、参考报告清单和
输出位置；运行后不得根据结果挑选或替换 seed。

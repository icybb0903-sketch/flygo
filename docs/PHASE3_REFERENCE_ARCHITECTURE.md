# 阶段 3 参考实现审计与落地架构

审计日期：2026-09-18（Asia/Shanghai）
审计范围：只读检查项目现状、官方 MaleCNS 资料、原始模型仓库，以及用户指定的 `sandbornm/housefly` 21 点实现。本文件不代表已经把完整 MaleCNS 接入围棋。

## 结论先行

1. **Google/Janelia 发布的是 MaleCNS v1.0 连接组数据，不是一个下载后即可下棋的 AI 或可执行“大脑”。** 官方数据提供神经元/分段标识、注释、预测神经递质、连接权重和形态数据；膜电位方程、刺激编码、任务读出和学习规则必须另外实现。
2. **`sandbornm/housefly` 当前版本确实有一条可执行的神经控制路径。** 它不是只有一段 3D 表演：浏览器 Worker 中运行一个 Rust/WASM LIF 模型，场景把观察编码成 32 个刺激率，模型产生真实的模拟帧和 128 个输出池率，线性读出再在合法动作中选择动作。动画视图复制同一模型帧的电位和脉冲。
3. **它仍然不是“生物果蝇已经学会 21 点”的证据。** 其连接权重被冻结；视觉输入组、输出池、神经递质符号、动力学常数、读出和奖励学习均含工程选择。仓库本身也明确写着“experimental neural control, not validated fly cognition”。
4. **21 点策略不能直接复用到 9×9 围棋。** 21 点只有 4 个动作和短回合奖励；本项目固定为 82 个动作（81 个交叉点加停一手），状态空间、合法性、长期信用分配和训练数据完全不同。可以复用的是运行时和接口纪律，而不是 Blackjack 的编码器、权重或策略结论。
5. **推荐阶段 3 先完成“真实 MaleCNS 帧可运行、可验证、可显示”，再在阶段 4 接管围棋落点。** 阶段 3 未训练读出不应偷偷调用当前 capture-first 基线；无活动、资产损坏、取消或运行错误时必须停止神经动作并如实显示错误。

## 固定上游版本与证据

### 官方 MaleCNS

- 官方下载页：<https://male-cns.janelia.org/download/>
- 数据集：`male-cns:v1.0` / MaleCNS v1.0
- 官方页面说明连接权重 Feather 是完整的 segment-to-segment connection graph，并提供注释、神经递质预测、突触位置、骨架和 neuPrint 访问。
- 官方 MaleCNS 页面声明数据采用 CC BY；本项目已有审计将其固定为 CC BY 4.0。
- 官方 Google Research 概览报告该连接组覆盖超过 166,000 个神经元和约 1.25 亿个突触连接：<https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/>
- 原始论文/预印本：<https://doi.org/10.1101/2025.10.09.680999>

本项目本地已有并校验的三个 v1.0 文件：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |

这些哈希与本项目 `docs/P0_FEASIBILITY.md`、完整下载清单以及 Housefly 的运行时清单相符。它们证明数据来源一致，不证明任何动力学或围棋能力。

### Housefly 21 点实现

- 仓库：<https://github.com/sandbornm/housefly>
- 本次审计固定提交：[`2a15c83f9e10a924ff4da7f49aec07463f5c65cf`](https://github.com/sandbornm/housefly/tree/2a15c83f9e10a924ff4da7f49aec07463f5c65cf)
- 提交时间：2026-09-14T02:18:08Z
- 截至审计时仓库没有 GitHub release，也没有可用版本标签；因此不能依赖浮动的 `main`，必须记录完整提交哈希。
- 原创代码许可证：MIT；仓库中的 MaleCNS 数据仍是 CC BY，NeuroMechFly v2 身体网格仍是 Apache-2.0。

审计时发现一个重要文档差异：固定提交的根 `README.md` 和实际 `src/main.ts` 已提供默认的 `Neural LIF` 模式，也保留可切换的 `Odds baseline`；但 `docs/architecture.md` 顶部仍描述较早的“数学策略 + illustrative diffusion”版本。因此，判断当前实现应以固定提交的代码、根 README、`neural/README.md` 和清单为准，不能单独引用旧架构页。

### LIF 模型参考

- 原始研究代码：<https://github.com/philshiu/Drosophila_brain_model>
- Housefly 固定的上游提交：[`91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`](https://github.com/philshiu/Drosophila_brain_model/tree/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960)
- 代码许可证：MIT
- 论文：<https://doi.org/10.1038/s41586-024-07763-9>

该原始模型针对 FlyWire 雌性全脑版本构建。Housefly 把其动力学思想移植到 MaleCNS，并加入自己的节点筛选、符号映射、Rust/WASM 实现和任务接口；这属于第三方工程适配，不能描述成 MaleCNS 官方提供的执行模型。

## Housefly 的真实结构

固定提交的主路径是：

```text
可观察的游戏状态
  -> 任务专用 encode()：32 个 0..150 Hz 工程刺激率
  -> Rust/WASM LIF：冻结的 MaleCNS 稀疏图
  -> NeuralFrame：电位、脉冲、计数、128 个输出池率
  -> 线性 readout：动作 logits/概率
  -> 仅规则合法性 mask
  -> 实际执行动作
  -> 可选 teach/reinforce 只更新 readout
```

关键文件与职责：

| 上游文件 | 已核实作用 | 对本项目的价值 |
|---|---|---|
| `neural/scripts/build_graph.py` | 读取三张 Feather、校验来源、选择节点/边、生成 CSR 式分片和清单 | 可参考数据列选择、源文件哈希、确定性排序和产物清单 |
| `neural/src/lib.rs` | 事件驱动的稀疏 LIF、延迟传播、 refractory、状态快照 | 可复用算法或作独立 Python 实现的数值对照 |
| `src/neural/data.ts` / `runtime.ts` | 校验并加载 WASM/图资产，暴露同步运行时 | 可参考资产 fail-closed 和帧结构 |
| `src/neural/worker.ts` / `client.ts` | 把重计算移出渲染线程，限定队列、取消旧请求 | 可参考本项目计算任务的取消、超时和过期帧防护 |
| `src/neural/task.ts` | 固定模拟窗口；编码、合法动作和读出接口 | 可直接借鉴任务适配契约 |
| `src/neural/policy.ts` | 只读 128 个神经率的线性读出；支持监督和策略梯度更新 | 可借鉴 82 动作读出与检查点格式，不能复用 Blackjack 权重 |
| `src/activities/blackjackNeural.ts` | 21 点 32 通道编码、4 动作合法 mask、120 ms 决策窗口 | 只作集成范例，不能直接用于围棋 |
| `src/connectomeView.ts` / `activityConnectome.ts` | 真实帧电位/脉冲复制到点云；显示边只是结构子集 | 可参考“控制全图与显示子集分离、但活动帧一致”的原则 |
| `public/neural/manifest.json` | 记录来源、筛选、参数、分片哈希和模型 ID | 可参考本项目运行时清单格式 |

Housefly 的运行图不是官方 166,691 神经元的无损“整脑模拟”。它明确筛选为 139,662 个有分类且有 soma 坐标的节点，并保留权重至少 5 的 5,536,347 条边。其界面另外只画 60,000 条结构边，而控制器仍计算 5,536,347 条边。这个区分很适合本项目，但必须显示“模型保留数”和“官方数据规模”，不能都写成“完整”。

Housefly 的当前 LIF 清单记录了 0.2 ms 步长、-52 mV 静息/复位、-45 mV 阈值、1.8 ms 延迟、2.2 ms refractory、每 contact 0.275 mV 等参数。它还把 ACh 映射为正，GABA/谷氨酸/组胺映射为负，调质/不明映射为零。仓库明确承认这是未验证的简化符号假设；特别是谷氨酸并非总是抑制，零快速电流也不等于这些调质连接在生物上没有作用。

## 不能从 21 点实现直接带到围棋的内容

| 21 点实现 | 为什么不能直接复用 | 围棋所需替代 |
|---|---|---|
| `blackjack-observation-32-v1` | 输入包含牌点、庄家明牌和剩余牌型，不含棋盘空间结构 | 从本项目 415 维确定性围棋编码再映射为版本化的 32 通道刺激率 |
| `Hit/Stand/Double/Split` 4 动作读出 | 与 82 个围棋动作无对应关系 | 固定 `0..80` 行优先交叉点 + `81` 停一手 |
| 浏览器 localStorage 中的 Blackjack 权重 | 动作顺序、编码器和任务都不同 | 新的 Go readout 检查点，绑定模型 ID、编码器版本、动作顺序和训练数据哈希 |
| 一局 Blackjack 的即时输赢奖励 | 围棋回合更长、奖励稀疏且信用分配困难 | 阶段 5 的离线教师样本和/或经审计的整局强化学习 |
| 精确赔率基线 | 它是另一个控制器，不经过神经帧 | 只能离线产生训练/评估目标；MaleCNS 运行模式绝不能在决策时调用它 |
| 120 ms 决策窗口 | 这是上游任务参数，不是生物定律 | 在本机 benchmark 后固定围棋窗口并写入模型清单 |
| 128 池随机初始线性读出 | 冷启动只是随机策略，不能称为学会 | 先标为未训练；达到阶段 5 验收后才加载训练检查点 |

## 本项目阶段 3 推荐架构

### 1. 数据准备层

新增一个离线、确定性的预处理命令，从已经校验的三个 Feather 构建只读运行资产：

```text
data/malecns/v1.0/*.feather
  -> 验证固定字节数和 SHA-256
  -> 选择并排序模型节点
  -> 映射 bodyId <-> 连续 index
  -> 构建 outgoing CSR：offsets / targets / contact weights / signs
  -> 导出 soma 坐标与显示边子集
  -> 生成 manifest.json（来源、选择规则、计数、参数、每文件哈希）
```

建议阶段 3 第一版严格复现 Housefly 的可审计选择规则，便于逐项比较：

- 节点：`superclass` 非空且有 `somaLocation`；按数值 body ID 排序。
- 边：两端都保留，contact weight `>= 5`。
- 快速符号：ACh `+1`；GABA/glutamate/histamine `-1`；其余 `0`，并醒目标为未验证假设。
- 模型 ID 必须包括数据版本、选择规则、动力学版本和构建器版本。

如果本项目最终采用不同节点或阈值，必须生成不同模型 ID 和自己的统计；不得继续引用 Housefly 的 139,662/5,536,347 数字。

运行时资产建议放在 `data/malecns/runtime/<model-id>/`，不要覆盖三张原始 Feather。原始文件、准备产物和显示产物分别记录哈希。

### 2. 围棋刺激适配层

本项目已有 `go-neural-encoding-v1` 风格的固定 415 维状态记录：棋子平面、空位、合法点、最后一手和标量。阶段 3 再增加单独的、版本化的刺激适配器：

```text
GoState -> 415 维记录 -> 32 个有限刺激率（0..150 Hz）
```

纪律：

- 只能读取对局中公开且因果上已经发生的状态。
- 不得写入教师建议、基线选点、下一手答案或隐藏搜索值。
- 对称变换、通道分组、归一化和压缩矩阵全部写入配置并参与哈希。
- 同一状态、模型版本和种子必须产生相同输入率。
- 如果 32 通道压缩损失太大，可以在以后发布新的编码器版本；不能静默改变含义。

### 3. 稀疏神经运行时

推荐先在 Python 侧实现可验证参考运行时，保持现有本地服务结构；如性能不够，再把同一模型迁到 Housefly 式 Rust/WASM Worker。两条路径必须用相同的小图数值 fixture 对齐。

运行时最少输出：

- `model_id`、数据/构建清单哈希、随机种子；
- 模拟 tick、模拟毫秒和真实墙钟延迟；
- 本窗口真实产生的 spike index/count；
- 用于显示节点的真实模型电位或有界 level；
- 128 个只从未直接驱动输出池脉冲统计得到的率；
- `silenced`、错误和取消状态。

不要把当前 `src/toy_dynamics.py` 的 12 节点、单跳工程电位直接升级标签为“完整 MaleCNS”。它应继续保留为小型回归 fixture；完整运行时必须有新的模块名、模型 ID 和清单。

### 4. 82 动作读出

阶段 3 可以建立未训练读出接口，但不应宣称具备棋力：

```text
128 个真实输出池率
  -> 固定维度线性层（82 x 128，可加 bias）
  -> 82 个原始 logits
  -> 围棋规则引擎产生的 82 位 legal mask
  -> 合法集合内的概率/argmax
```

合法 mask 只排除规则非法动作，不能编码“策略上不喜欢”的点。若所有神经率为零、模型静默、资产损坏、读出不兼容或请求被取消，结果应为 `withheld`；不得调用 capture-first、随机或搜索程序代替。

检查点至少绑定：

- `model_id`
- 原始数据与运行时资产清单哈希
- 围棋编码器版本和刺激适配器版本
- 82 个动作的固定顺序
- readout 架构、权重哈希、训练算法和训练数据/教师版本
- 训练随机种子与创建时间

### 5. 同源神经动画

右侧动画不需要画全部 553 万条边。可以画所有有位置节点和固定的 60,000 条结构显示边，或在显存/帧率测试后使用更小的确定性子集；但必须区分：

- `controller_node_count` / `controller_edge_count`：参与实际计算；
- `display_node_count` / `display_edge_count`：用于浏览器绘制；
- `activity_source=controller_frame`：点的亮度和 spike 标记来自控制该动作的同一真实帧。

不允许另起一个扩散动画冒充控制器活动。装饰性沿边粒子如果保留，必须明确标成装饰并不得作为 spike。每次决策的棋盘、果蝇动作和右侧帧应共享：

```text
decision_id + board_hash + encoding_hash + model_id + frame_id + checkpoint_hash
```

### 6. API 与并发边界

现有 `ThreadingHTTPServer` 可以继续承载静态页面和小请求，但一次完整神经计算应进入单独的有界任务队列，避免两个请求并发修改同一脑状态。

建议接口：

- `GET /api/neural/status`：只报告已加载模型、清单哈希、计数、运行时状态和最近错误。
- `POST /api/neural/frame`：输入版本化的围棋状态/编码哈希，返回真实神经帧；阶段 3 不执行棋步。
- 阶段 4 再提供 `POST /api/go/neural-move`：由同一帧读出、合法过滤并提交动作。

请求应有 `request_id`/`generation`。悔棋、重开、暂停或新棋局必须让旧任务过期；过期结果可记录但不得落子。队列有界，只允许每个脑实例一个计算进行中。超时、异常、源清单不符和产物损坏都要 fail closed。

## 阶段 3 验收门

在进入“MaleCNS 决定围棋落点”的阶段 4 前，至少满足：

1. 三张原始 Feather 的字节数和 SHA-256 与固定清单一致。
2. 预处理重复两次得到同一产物哈希、节点顺序和边顺序。
3. 运行图计数与选择规则相符；未知神经递质、缺失坐标和零符号连接都有显式统计。
4. 小型正/负连接 fixture 与独立参考实现对齐，包括延迟、复位和 refractory 边界。
5. 固定模型、输入、种子和分块方式可确定性重放；如果实现使用随机 Poisson 刺激，随机生成器和种子必须固定。
6. 零输入静默测试、silence 测试、零边测试、断开边对照和损坏资产拒绝均通过。
7. 在本机记录峰值内存、预处理时间、每 20/100/120 ms 模拟窗口墙钟耗时；先测量再决定最终窗口。
8. 浏览器显示的 spike/level 与 API 同一 `frame_id` 的原始数组抽样一致。
9. 阶段 3 页面仍标为“神经传播已接入，围棋决策尚未接管”或等价真实状态。
10. 运行错误时页面明确显示“不可用/动作停止”，不切换隐藏基线。

Housefly 固定提交给出的约 46 MB 浏览器资产和其 macOS/Node 性能数据只能当上游参考，不能当作本机验收结果。本项目必须在 E 盘实际构建并测量。

## 许可证与署名清单

### MaleCNS 数据

MaleCNS v1.0 数据采用 CC BY 4.0。发布页面、截图、视频、模型产物或可下载派生数据时至少应：

- 清楚标注数据来自 HHMI Janelia MaleCNS v1.0，并列出官方合作方/官方来源链接；
- 链接 CC BY 4.0；
- 说明本项目做过哪些筛选、阈值化、坐标归一化、符号映射和格式转换；
- 不让项目代码许可证覆盖或误写数据许可证。

### Housefly 代码

Housefly 原创代码是 MIT。若复制或实质改写其 `build_graph.py`、Rust/WASM 运行时、Worker、任务控制器、读出或可视化代码，应在项目中保留 Mark Sandborn 的 MIT 版权/许可文本，记录固定提交和修改内容。仅借鉴接口思想也应在本文件/技术说明中署名，但不应把我们的实现描述成 Housefly 官方分支。

### Shiu/Spiller 模型代码

`Drosophila_brain_model` 代码是 MIT。若复制公式实现、调度代码或 fixture，应保留 Philip Shiu 和 Nico Spiller 的 MIT 通知并引用论文。引用模型参数不等于证明对 MaleCNS 的生物校准有效。

### 3D 资产

Housefly 的 NeuroMechFly v2 身体网格按 Apache-2.0 提供。如果本项目以后直接采用该 GLB，需要同时加入 Apache-2.0 NOTICE/许可证和来源说明。当前项目程序化果蝇不需要为了“像 Housefly”而复制该网格。Three.js 的现有 MIT 许可证应继续保留。

## 表述红线

可以说：

> “这是使用 MaleCNS v1.0 实测结构连接、第三方/本项目工程 LIF 动力学和任务读出的实验性围棋控制器。”

不应说：

- “Google 开源了会下棋的果蝇大脑。”
- “这就是完整数字果蝇，行为来自生物记忆。”
- “右侧亮点证明真实果蝇正在思考这一手。”
- “接入 MaleCNS 就说明策略优于普通 AI。”

## 给主 Agent 的执行建议

阶段 3 的最短诚实路线是：

1. 先固定 Housefly 提交和许可，不下载其媒体或大资产。
2. 用本地三张已验证 Feather 构建本项目自己的确定性稀疏运行资产和清单。
3. 用小图 fixture 验证 LIF 后，再跑完整图的性能门；不要先接 UI 再补真实性。
4. 把现有围棋 415 维记录映射到版本化的 32 通道刺激，不包含教师动作。
5. 返回真实神经帧并让右侧显示同一帧；此时仍不接管白棋。
6. 通过上述验收后，阶段 4 才添加 128 -> 82 读出并替换 capture-first 控制器。

这条路线复用了 Housefly 最有价值的部分——全计算图与显示子集分离、同一真实帧驱动读出与动画、合法 mask、取消/静默 fail-closed、固定连接组只训练读出——同时避免把 21 点策略、演示视频或未验证的生物结论冒充围棋能力。


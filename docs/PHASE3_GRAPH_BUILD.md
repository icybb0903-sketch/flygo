# 阶段 3：MaleCNS 运行图构建

本阶段把本地三份 MaleCNS v1.0 Feather 转成可内存映射的 CSR 图。它使用真实连接记录，但节点筛选、权重阈值和神经递质符号都是工程选择，**不是已经验证的生物仿真**。

## 固定规则

- 源文件必须逐字节匹配代码内固定的 MaleCNS v1.0 SHA-256，否则停止构建。
- 节点：`superclass` 非空，且 `somaLocation` 恰为三个有限数值；按数值 `bodyId` 升序。
- 边：原始记录两端都在节点集，且原始 `weight >= 5`。
- 边按 `(pre dense index, post dense index)` 排序；重复 pair 的原始整数权重求和后再转为 `float32`。
- CSR 方向是突触前神经元行到突触后神经元列（pre -> post）。
- 神经递质读取 `consensus_nt`：乙酰胆碱 `1`；GABA、谷氨酸、组胺 `-1`；多巴胺、章鱼胺、血清素 `0`；缺失、`unclear` 或无法识别为 `-128`。

其中正负号与调质型的零快速电流是参考 Housefly 的**未验证工程假设**。特别是谷氨酸不应在生物学上被理解为永远抑制，`0` 也不表示调质连接没有生物功能。

## 文件契约

每个 `data/malecns/runtime/<model-id>/` 目录包含：

| 文件 | dtype | shape | 含义 |
|---|---|---|---|
| `node_ids.npy` | `int64` | `[N]` | dense index 到原始 body ID |
| `soma_xyz.npy` | `float32` | `[N,3]` | soma 坐标 |
| `indptr.npy` | `uint64` | `[N+1]` | outgoing CSR 行指针 |
| `indices.npy` | `uint32` | `[E]` | post/destination dense index |
| `weights.npy` | `float32` | `[E]` | 重复 pair 合并后的接触权重 |
| `nt_code.npy` | `int8` | `[N]` | 每个 presynaptic 节点的工程符号码 |
| `manifest.json` | JSON | — | 来源、规则、计数、shape/dtype/hash、模型 ID |

数组是独立未压缩 `.npy`，加载器用 `numpy.load(..., mmap_mode="r")`。加载时复核 manifest 模型 ID、全部产物 SHA-256、dtype、shape、节点顺序、CSR 单调性、索引范围、正权重和 NT code 范围。任一错误均 fail closed。

`model_id` 来自不含 `model_id` 自身的 canonical manifest 内容 SHA-256，包括源哈希、筛选规则、真实计数和全部数组哈希。manifest 不含时间字段。构建先在同卷 `.staging-*` 目录完成并通过加载器复核，最后才原子改名发布；不会覆盖损坏或内容冲突的同名目录。

## 命令

必须使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe scripts\build_malecns_runtime.py
.\.venv\Scripts\python.exe -m unittest tests.test_malecns_assets -v
```

## 实际全量构建结果

2026-09-18 在本机使用项目 `.venv`（Python 3.12.14、NumPy 2.3.5、PyArrow 25.0.1）对默认三份 Feather 完成两次独立全量构建：

| 指标 | 实测值 |
|---|---:|
| 第一次构建墙钟时间 | 21.555 s |
| 第二次构建墙钟时间 | 22.537 s |
| 第二次进程采样峰值 working set | 1,194,889,216 B（约 1.11 GiB） |
| 原始权重记录 | 151,856,684 |
| 入选节点 | 139,662 |
| 过滤后记录 | 5,536,347 |
| distinct CSR 边 | 5,536,347 |
| 合并的重复记录 | 0 |
| 正式目录大小 | 48,345,390 B（约 46.11 MiB） |

峰值来自 Windows 进程 working set 每 200 ms 采样，是测得的近似上界，不是 Python heap 精确剖析。第二次构建写入独立输出根；两个输出的 model ID、manifest 和六个 `.npy` 文件 SHA-256 全部一致。

模型与清单：

- `model_id`: `malecns-v1.0-w5-3acb6434e71160fc`
- `manifest.json`: `73b436cf60a3773ea57fce007f53f50c97ebfc5d6d953047d38f7bd18cee54dd`

| 产物 | 字节数 | SHA-256 |
|---|---:|---|
| `node_ids.npy` | 1,117,424 | `0474fd89529b3e5cab726a46f39256acaa45d8fe5c0982ce9aecbb0498719a1d` |
| `soma_xyz.npy` | 1,676,072 | `4429f6365cba4378a51e5b137cb9900ddb76bce4498128edaf6cc12018103794` |
| `indptr.npy` | 1,117,432 | `69003d22a3e0a0404d283ba9a13f4d23846a004bfcf9f00978c313d7ed1e48ad` |
| `indices.npy` | 22,145,516 | `afdbecec587c7b7477e6db017d4947490a6afa4819545b6d9de9edaa1c8aaa0e` |
| `weights.npy` | 22,145,516 | `95b0dca3ba25517f5ce2988870b6395f5790c6432ba3455397d724529260beab` |
| `nt_code.npy` | 139,790 | `b3db620c9c5ee2570027f1192a6c4c3a1cb89d5d908a3c48d157a361f022a6f5` |

神经递质统计：139,652 个入选节点在 NT 表中有记录，10 个没有；最终 code 数为兴奋 85,847、抑制 51,053、调质/零快速电流 520、unknown 2,242。unknown 包含已有记录但 `consensus_nt=unclear` 的节点，因此大于“缺失记录”10。

测试：合成 fixture 的 4 项测试在 0.608 s 内通过，覆盖阈值、重复边合并、缺失 NT、mmap、源哈希拒绝、产物损坏拒绝和两次构建逐文件确定性。

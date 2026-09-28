# 阶段 3：MaleCNS 完整数据只读审计

审计日期：2026-09-18（Asia/Shanghai）

审计目录：项目内 `data/malecns/v1.0`
范围：只读取三份 MaleCNS Feather 文件、Python 环境与机器资源。本审计没有修改生产代码、测试或原始数据。

## 结论先行

1. 三份文件都是 Arrow IPC/Feather V2 文件（起始 magic 为 `ARROW1`），可用 `pyarrow.memory_map` 和 `pyarrow.ipc.open_file` 按 record batch 读取。
2. 权重表不是一个只有约 20 万个已注释细胞的小图：它有 **151,856,684 条连接记录**，端点联合覆盖 **88,384,522 个不同 body ID**。一次把全表及所有唯一 ID 装进 Python 对象并不适合运行时。
3. 注释表有 **211,577** 个唯一 `bodyId`；神经递质表有 **1,835,518** 个唯一 `body`。两者 ID 交集为 **187,016**。
4. 只保留“首尾都在注释表中”的连接后还有 **26,028,386 条连接记录**；再要求 `weight >= 5` 后为 **6,300,108 条记录**。这是当前机器上更适合实时传播的候选运行时子图。
5. 权重表没有按 `(body_pre, body_post)` 排序。此次没有冒险为 1.52 亿行做全量排序/去重，所以 **不同连接对的精确数量当前不可获取**。151,856,684 是记录数，不应被写成已证明的 distinct pair 数。预处理器必须在过滤后显式排序并合并重复键。

## 1. 文件真实性与完整性（实测）

下表的字节数及哈希是本次在 E 盘重新读取文件得到的实测值；目录中没有 `manifest.json`。

| 文件 | 字节数 | MD5 | SHA-256 |
|---|---:|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | `50A7718770C57220F160BA4F431AB89E` | `2177E246113E4CFBF1E7772EC37C6DA1955FF22E8063D0B1F833101F99A9A3B2` |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | `3D842B12FE5C49EEFADE528D7DD24A1F` | `95C9289220663ABEB3409F3AD9E5A7F8A53F8093F5139D15502CD08DA8879621` |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | `F30E9DCCA25CFD021BF1E7B3D975599E` | `E35DA783D1C686B2B58B3B87CD6A403AE43BFCFBA8BFF28E08EF752C1A56AFC1` |

校验命令：

```powershell
Get-ChildItem data\malecns\v1.0\*.feather | ForEach-Object {
  Get-FileHash -LiteralPath $_.FullName -Algorithm MD5
  Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
}
```

## 2. 文件格式、schema、行数和批次数（实测）

### 2.1 汇总

| 表 | 行数 | 列数 | Record batches | ID 列唯一数 | ID null |
|---|---:|---:|---:|---:|---:|
| annotations | 211,577 | 36 | 4 | 211,577 (`bodyId`) | 0 |
| neurotransmitters | 1,835,518 | 10 | 29 | 1,835,518 (`body`) | 0 |
| weights | 151,856,684 | 3 | 2,318 | 见下节 | 三列均为 0 |

正常批次均为 65,536 行；三个表最后一批分别为 14,969、510、9,772 行。

### 2.2 权重表 schema

```text
body_pre:  int64, nullable
body_post: int64, nullable
weight:    int64, nullable
```

全量流式统计：

| 指标 | 实测值 |
|---|---:|
| 连接记录数 | 151,856,684 |
| 不同 `body_pre` | 1,834,661 |
| 不同 `body_post` | 87,576,984 |
| 两端 body ID 联合唯一数 | 88,384,522 |
| `weight` 最小 / 最大 | 1 / 2,591 |
| `weight` 总和 | 311,833,243 |
| `(body_pre, body_post)` 是否按字典序排序 | 否 |
| 相邻重复 pair | 0（不能据此证明全表无非相邻重复） |
| distinct pair 精确数 | 不可获取：本次未做全量外排或全量哈希去重 |

### 2.3 神经递质表 schema

```text
body: int64
cell_type: string
total_nt_predictions: int32
predicted_nt_confidence: double
predicted_nt: string
ground_truth: string
celltype_total_nt_predictions: int32
celltype_predicted_nt: string
celltype_predicted_nt_confidence: double
consensus_nt: string
```

与传播相关的缺失情况：`body`、`predicted_nt`、`consensus_nt` 均为 0 null；`predicted_nt_confidence` 有 857 个 null；`cell_type` 有 1,671,072 个 null；`ground_truth` 有 1,750,034 个 null。因此运行时不能把 `cell_type` 或 `ground_truth` 当作每个 body 都存在的必填值。

### 2.4 注释表 schema

```text
assignedOlHex1: double              assignedOlHex2: double
bodyId: int64                       flywireType: string
group: double                       instance: string
somaSide: string                    statusLabel: dictionary<string, int8, ordered>
superclass: string                  type: string
vfbId: string                       hemibrainType: string
itoleeHl: string                    supertype: string
birthtime: string                   mancBodyid: double
mancGroup: double                   mancType: string
subclass: string                    synonyms: string
class: string                       rootSide: string
somaNeuromere: string               trumanHl: string
dimorphism: string                  matchingNotes: string
entryNerve: string                  mancSerial: double
mcnsSerial: double                  serialMotif: string
fruDsx: string                      exitNerve: string
receptorType: string                somaLocation: list<int64>
tosomaLocation: list<int64>         status: string
```

`bodyId` 无 null 且 211,577 行全部唯一。其余注释列允许大量缺失，预处理器应只把它们作为可选元数据。

## 3. 可批量与内存映射读取性（实测）

本次成功使用以下读取路径扫描完整权重表：

```python
with pyarrow.memory_map(path, "r") as source:
    reader = pyarrow.ipc.open_file(source)
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        # 每次只处理当前 batch
```

结论：

- **可内存映射**：Arrow IPC footer/schema 打开约 0.004--0.007 秒，不需要先把 1.05 GB 文件复制进 Python heap。
- **可批量读取**：权重表有 2,318 个批次，每批约 65,536 行，单批三列固定宽度原始数据约 1.5 MiB。
- **注意**：memory map 只解决文件访问；压缩批次仍要解压到 Arrow buffer。`memory_map` 不等于“整表不占内存”。
- **不要在运行时建立全量 Python `set`**：为了精确统计 8,838 万个不同 ID，本次审计进程峰值 working set 达到 12,739,960,832 字节（约 11.86 GiB）。这是审计统计成本，不是推荐实现。

### 全量扫描基准

| 工作负载 | 耗时 | 峰值/内存说明 |
|---|---:|---|
| weights：全部 2,318 批，统计行数、权重和、唯一端点、顺序 | 49.626 秒 | 进程峰值 working set 12,739,960,832 B；Arrow allocator 采样最大 1,572,864 B；高峰主要来自 8,838 万 ID 的 Python 集合 |
| annotations：全表行数、唯一 ID、各列 null | 0.3638 秒 | 该进程峰值 working set 159,084,544 B |
| neurotransmitters：全表行数、唯一 ID、各列 null | 0.2543 秒 | 同一进程累计峰值 working set 317,128,704 B |
| weights：两套 ID membership、注释诱导边、阈值计数 | 194.102 秒 | 低内存批处理；没有构造全量边表或全量端点集合 |

这些是单次本机测量，不应当作跨机器性能保证。

## 4. 子图规模（实测）

第二次完整流式扫描用排序后的 ID 数组和 `searchsorted` 做 membership，没有读取整表为单个 Arrow Table。

| 条件 | 连接记录数 |
|---|---:|
| 两端都在 annotations | 26,028,386 |
| 至少一端在 annotations | 143,878,870 |
| 两端都在 neurotransmitters | 32,789,858 |
| 至少一端在 neurotransmitters | 151,856,684 |
| annotations 诱导子图中实际出现在边上的注释 ID | 188,778 |
| annotations 与 neurotransmitters 的 ID 交集 | 187,016 |

annotations 诱导子图的权重阈值：

| 最小 `weight` | 连接记录数 | 相对未过滤注释诱导边 |
|---:|---:|---:|
| 1 | 26,028,386 | 100% |
| 2 | 15,514,634 | 59.607% |
| 3 | 10,653,945 | 40.932% |
| 5 | 6,300,108 | 24.205% |
| 10 | 2,769,379 | 10.640% |
| 20 | 1,067,386 | 4.101% |
| 50 | 228,209 | 0.877% |
| 100 | 55,515 | 0.213% |

这里仍是“记录数”，不是已去重后的 pair 数。

## 5. 当前 Python 依赖与机器约束（实测快照）

项目虚拟环境 `.venv`：

| 项目 | 实测值 |
|---|---|
| Python | 3.12.14 |
| pyarrow | 25.0.1 |
| numpy | 2.3.5 |
| pandas | 3.0.1 |
| scipy | 未安装 |
| polars | 未安装 |

系统默认 `C:\Python314\python.exe` 是 Python 3.14.6，且本次检查时 `pyarrow`、`numpy`、`scipy`、`polars`、`pandas` 均不可导入。阶段 3 命令必须显式使用项目 `.venv\Scripts\python.exe`，不能依赖 `python` 命令碰巧指向正确环境。

机器资源快照：

| 资源 | 实测值 |
|---|---:|
| CPU | AMD Ryzen 5 7500F，6 核 / 12 逻辑处理器 |
| 物理内存 | 33,980,137,472 B（31.65 GiB） |
| 检查时可用物理内存 | 22,481,379,328 B（20.94 GiB） |
| 虚拟内存 | 36,127,621,120 B（33.65 GiB） |
| E 盘可用空间 | 151,406,710,784 B（141.01 GiB） |

资源值会随其他进程和文件变化，它们只是审计时快照。

## 6. 阶段 3 预处理产物建议

### 6.1 推荐的两个产物层级

**A. 可追溯的注释诱导图（离线/验收）**

- 节点集合：annotations 的 211,577 个 `bodyId`。
- 边：首尾均在节点集合中的记录，先按 `(body_pre, body_post)` 排序，再对重复 pair 求和。
- 不先加权重阈值，保留用于审计的完整注释诱导图。
- 每个产物写 manifest：源文件 SHA-256、过滤条件、输入记录数、输出 distinct pair 数、重复合并数、节点数、dtype、生成器版本和生成时间。

**B. 实时传播图（运行时）**

- 初始候选规则：A 的基础上使用 `weight >= 5`。
- 此阈值是为了工程实时性的**建议值**，不是数据发布方规定，也不是已证明的生物学最佳阈值。
- 预处理完成后必须用真实传播 benchmark 决定保留 `>=5`，还是改用 `>=10` 或其他阈值；不得只按文件大小判断。

### 6.2 建议文件格式

不要把运行时 CSR 存成一个需要整体解压的压缩 `.npz`。建议使用可 `numpy.load(..., mmap_mode="r")` 的独立 `.npy` 数组：

```text
artifacts/malecns/runtime_w5/
  node_ids.npy       int64    dense index -> original body ID
  indptr.npy         uint64   CSR row pointers
  indices.npy        uint32   CSR destination dense indices
  weights.npy        float32  merged edge weight（或文档化后的归一化值）
  nt_code.npy        int8     neurotransmitter/sign code，未知值保留显式 sentinel
  annotations.arrow           仅保留展示所需的少量注释列
  manifest.json               来源哈希、规则、统计、dtype 和产物哈希
```

理由：

- `.npy` 是简单稳定的连续数组，可真正 memory-map；
- 88,384,522 < 2^32，dense destination index 可安全使用 `uint32`；
- `indptr` 使用 `uint64`，避免未来边数扩展造成溢出；
- 原始单行 `weight` 虽然最大仅 2,591，但重复 pair 合并后的上限本次尚未测出，不能先假定 `uint16` 一定安全；`float32` 也便于传播归一化；
- `nt_code` 必须有 unknown，不能用缺失值静默假装兴奋或抑制。

### 6.3 大小估算（明确为估算）

以下按“未去重记录数作为边数上界”计算；排序合并重复 pair 后只会相同或更小，不包含 `.npy` 小量 header、JSON、可选注释字符串和临时排序空间。

| 候选产物 | 节点上界 | 边记录上界 | 数组大小估算 |
|---|---:|---:|---:|
| 全量 8,838 万 body 图；`int64 ids + uint64 indptr + uint32 indices + float32 weights + int8 nt` | 88,384,522 | 151,856,684 | 2,717,390,354 B（约 2.53 GiB） |
| annotations 全诱导图 | 211,577 | 26,028,386 | 211,823,905 B（约 202.01 MiB） |
| annotations 且 `weight >= 5` | 211,577 | 6,300,108 | 53,997,681 B（约 51.50 MiB） |
| annotations 且 `weight >= 10` | 211,577 | 2,769,379 | 25,751,849 B（约 24.56 MiB） |

`weight >= 5` 运行时图的两个 `float32` 活动向量另需约 1,692,616 B（1.61 MiB）。真正的传播时间、临时 scatter/reduction 内存和浏览器下采样大小尚未测量，不能从上述数组字节数直接推断。

### 6.4 安全的预处理顺序

1. 验证三份源文件的固定大小和 SHA-256。
2. 从 annotations 生成排序且唯一的允许节点 ID 数组。
3. 按 65,536 行批量扫 weights，先过滤“两端都在允许节点集”；将候选边写入临时 Arrow IPC/Parquet 分片，避免 Python 对象列表。
4. 对已缩小到约 2603 万条（或阈值后约 630 万条）的候选边排序，按 pair 聚合并检查求和后 dtype 范围。
5. 建 dense ID 映射与 CSR 独立 `.npy` 数组，原子发布到新目录。
6. 从 neurotransmitters 左连接 `nt_code`，对缺失和冲突做显式计数。
7. 重新读取产物，验证 CSR 单调性、索引范围、边数、哈希、确定性以及至少一轮传播；验证通过前不让网页宣称“MaleCNS 已接入”。

临时空间预算至少预留 **源文件 + 过滤分片 + 排序工作区 + 最终产物**。E 盘当前 141.01 GiB 可用，空间足够；仍应把临时目录限制在项目明确的 staging 子目录，并在成功发布后由主流程决定是否删除。

## 7. 本次命令与方法记录

环境与资源：

```powershell
.venv\Scripts\python.exe -c "import pyarrow,numpy,pandas; ..."
Get-PSDrive E
Get-CimInstance Win32_OperatingSystem
Get-CimInstance Win32_Processor
```

Arrow 元数据和流式扫描均从项目根目录用项目虚拟环境执行：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
@'
# pyarrow.memory_map + pyarrow.ipc.open_file
# 逐 reader.get_batch(i) 统计；没有调用 read_table 读取完整 weights
'@ | .venv\Scripts\python.exe -
```

唯一端点扫描用每批 `numpy.unique` 后更新集合；这解释了 11.86 GiB 峰值。子图扫描改用已排序 ID 数组和 `numpy.searchsorted`，没有建立全量边或全量端点集合。

## 8. 阶段 3 的硬性门槛

- 不把 151,856,684 条记录伪称为已验证的 distinct connections。
- 不在请求处理或网页线程内读取原始 1.05 GB Feather。
- 不把 8,838 万 body 的全图作为默认实时图，除非先给出真实传播延迟和内存证据。
- 预处理输出必须可复现、带源文件/产物哈希，且明确数据版本、节点过滤和权重阈值。
- 页面只能显示实际传播使用的节点/边数；另列“原始表记录数”，不能混为一谈。
- 右侧神经动画只可使用当前 decision ID 对应的真实活动向量；为了显示性能而下采样时，必须同时标注“计算图规模”和“显示节点/边规模”。

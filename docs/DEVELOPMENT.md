# flygo 开发与评测

## 发布包和默认策略

默认使用 `data/checkpoints/stage4/` 中的已训练线性读出，不自动选择“最新”候选。
运行图、检查点及其绑定哈希保持原样。发布包不包含本机训练缓存、私人日志和
历史整局报告；界面评测字段因此可能显示不可获取，不应补写未经验证的数字。

训练发生在外部输出层，连接组权重固定。默认网页对弈不做在线更新。

## 安装开发依赖与测试

在项目根目录，用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-phase3.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

源码保留共享后端的早期实验模块与回归测试。普通使用者运行
`scripts/run_flygo.py`：该入口只开放围棋及其神经 API，首页直接进入围棋，
不开放实验台、Agent Monitor 或旧规则 baseline 的落子命令。

## 新训练实验

下面是重新生成离线数据并训练一个新输出层的示例，**不是快速试玩步骤**。
它会进行大量神经模拟，耗时取决于机器；不要覆盖发布的默认检查点。

```powershell
.\.venv\Scripts\python.exe scripts\train_neural_go_readout.py --cache data\training\my-experiment --checkpoint data\checkpoints\candidates\my-experiment --games 8 --max-moves 72 --seed 20261001
```

训练脚本使用 capture-first 教师生成离线监督标签；运行时检查点只有线性参数、
来源与契约，不访问教师。它不是完整强化学习训练，也不是生物突触学习。

预览新检查点：

```powershell
.\.venv\Scripts\python.exe scripts\run_flygo.py --port 8001 --checkpoint data\checkpoints\candidates\my-experiment
```

更换检查点不会自动证明棋力提升。既有候选脚本可能需要相应训练缓存或对局
报告，发布包不会伪造这些输入；先使用脚本 `--help` 明确文件与选项。

## 评测不能只看训练一致率

- 区分训练拟合、独立局面与完整对局。
- 按完整棋盘状态和历史排除数据重叠，不只检查 seed。
- 冻结检查点后，用新种子比较合法随机基线与同输入神经信号置零对照。
- 报告双方颜色、自然终局比例、手数上限裁决、失败与耗时。
- 更高教师一致率不是“果蝇理解围棋”，动画也不是有效性证据。

参见 [正式评测设计](STAGE7_EVALUATION_PLAN.md)、
[神经策略契约](STAGE4_NEURAL_POLICY.md) 和 [棋盘编码](PHASE3_GO_ENCODING.md)。
这些阶段文档保留开发背景；其中历史成绩不等于当前发布检查点的独立棋力保证。

## 从原始数据重建

普通用户不需要重建。开发者可以安装 `requirements-phase3.txt`，
用 `scripts/download_malecns_full.py --help` 查看官方数据下载方式，
再运行 `scripts/build_malecns_runtime.py`。这需要下载约 1.1 GB 原始表，
并占用比运行图更多的构建内存。

严格保持 manifest、运行图、输出池与检查点绑定；不要修改哈希绕过校验。

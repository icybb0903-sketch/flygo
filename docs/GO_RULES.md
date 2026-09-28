# 9×9 围棋环境规则

`src/go_engine.py` 是 MaleCNS 策略连接组可调用的确定性外部环境。它只实现围棋状态转移，不声称果蝇连接组本身理解围棋，也不在规则层替策略挑选落子。

## 公共接口

- `initial_state(komi=7.5)` / `new_game(...)`：返回空棋盘，黑先。
- `legal_moves(state)`：按 `(row, column)` 从 `(0, 0)` 到 `(8, 8)` 的行优先顺序返回合法着点，最后一个动作 `None` 表示 pass。终局返回空元组。
- `step(state, move)`：应用一个坐标、`None` 或字符串 `"pass"`，返回新的 `GoState`。原状态不可变；非法动作抛出 `IllegalMoveError`。
- `score_area(state)`：返回中国数子法面积得分。
- `serialize_state(state)` / `deserialize_state(payload)`：稳定 JSON 往返，包含完整棋盘历史。
- `GoEngine`：给闭环调用方使用的薄状态封装，提供同名 `step`、`legal_moves`、`score` 和 `reset`。

坐标从零开始，第一项为行、第二项为列。棋子编码为 `EMPTY=0`、`BLACK=1`、`WHITE=2`。

## 已实现规则

1. 棋盘固定为 9×9，黑白交替落子。
2. 落在棋盘外或已有棋子的点非法。
3. 落子后先移除无气的相邻对方棋串；一次可提取多串、多子。
4. 提子完成后己方棋串仍无气，则判为自杀并拒绝。
5. 普通落子若重现历史上任一完整棋盘局面，则按“位置超级劫”拒绝。历史保存完整 81 点元组而不是哈希，因此没有哈希碰撞风险。pass 允许重复棋盘。
6. 一方 pass 后若对方落子，连续 pass 计数清零；连续两次 pass 后终局，不能再落子。
7. 终局采用中国面积计分：每方得分为盘上己方棋子数加仅与己方棋子接壤的空点区域数；白方另加 `komi`（默认 7.5）。与双方都接壤或不接壤的区域为中立点。

规则层不做死子协商或自动死活判断。调用方应在双方认为局面结束时 pass；如果盘上仍留有形式上存活的死子，面积计分会按当前盘面处理。这一限制是明确且确定性的，适合机器策略实验与可复现实验记录。

## 状态与非法动作保证

`GoState` 是冻结的数据类，其棋盘和历史均为元组。`step` 在所有合法性检查完成后才创建新状态。状态式 `GoEngine.step` 也只在函数式 `step` 成功后替换内部状态。因此占位、自杀、劫重复、越界、错误动作格式及终局后动作都不会改变环境状态。

序列化格式标记为 `go-state-v1`，保存棋盘、行棋方、双方提子数、连续 pass 数、手数、贴目以及每一手后的完整棋盘历史。反序列化会检查尺寸、点值、计数、历史长度和最终历史棋盘的一致性。

## 策略接入示例

```python
from src.go_engine import GoEngine

environment = GoEngine()
choices = environment.legal_moves()
chosen = choices[0]  # 由 MaleCNS 读出映射后的策略选择决定
next_state = environment.step(chosen)

if next_state.game_over:
    result = environment.score().to_dict()
```

不要把 `legal_moves` 的序号当作长期稳定的动作编码，因为非法点会被过滤。策略输出若使用固定 82 维，可约定 `row * 9 + column` 对应 0–80，81 对应 pass，再转换为上述坐标或 `None`。

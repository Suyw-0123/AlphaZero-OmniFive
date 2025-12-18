# Zero Padding Edge Detection

## 修改說明

這個資料夾包含了實現 Zero Padding 邊緣偵測功能的修改檔案與其相容的現成模型。

### 注意事項

⚠️ **模型不相容警告**：
- 這個修改會改變神經網路的輸入維度（4 → 5 通道）
- 舊的 `.model` 檔案無法載入
- 必須從零開始訓練新模型 (目前的現成模型在資料夾內，參數調整方法一樣[here](../description_for_human_play.md) )
- 目前性能比較好的模型:
  -  [best_policy_1050](best_policy_11_11_5_256x6/best_policy_1050.model)
  -  [best_policy_1450](best_policy_11_11_5_256x6/best_policy_1450.model)
  -  遊玩參數設定: n_playout = 1000, c_puct = 5
  -  這兩個模型強度是比常規方法的模型要強的

### 使用方式

1. 備份原本的 `game.py` 和 `policy_value_net_pytorch.py`
2. 用這個資料夾中的檔案替換原本的檔案
3. **重要**：由於輸入維度改變，需要重新訓練模型（舊模型不相容）

### 修改的檔案

#### 1. `game.py`
- `current_state()` 函數：
  - 特徵平面從 4 個增加到 5 個
  - 新增 Plane 4：邊緣標記（邊緣位置為 1，內部為 0）

#### 2. `policy_value_net_pytorch.py`
- `Net` 類別：
  - `conv_initial` 輸入通道從 4 改為 5
- `PolicyValueNet` 類別：
  - `policy_value_fn()` 中的 reshape 從 4 改為 5

### 問題描述
原本的 AI 無法正確識別棋盤邊緣的威脅（如邊緣的活三、活四），因為 CNN 的 `padding=same` 會在邊緣補零，使得網路難以區分「真正的邊界」和「內部空位」。

### 解決方案：第 5 個特徵平面（Edge Markers）

新增第 5 個特徵平面，明確標記棋盤邊緣位置：

```
對於 9x9 棋盤，邊緣標記平面如下：
1 1 1 1 1 1 1 1 1
1 0 0 0 0 0 0 0 1
1 0 0 0 0 0 0 0 1
1 0 0 0 0 0 0 0 1
1 0 0 0 0 0 0 0 1
1 0 0 0 0 0 0 0 1
1 0 0 0 0 0 0 0 1
1 0 0 0 0 0 0 0 1
1 1 1 1 1 1 1 1 1
```

### 特徵平面定義

| 平面 | 描述 |
|------|------|
| 0 | 當前玩家的棋子位置 |
| 1 | 對手的棋子位置 |
| 2 | 上一步落子位置 |
| 3 | 顏色指示器（先手為全 1） |
| 4 | **邊緣標記**（邊緣為 1，內部為 0）【新增】 |



### 驗證修改

可以用以下程式碼驗證邊緣標記平面：

```python
from game import Board

board = Board(width=9, height=9, n_in_row=5)
board.init_board()
state = board.current_state()

print(f"State shape: {state.shape}")  # 應該是 (5, 9, 9)
print(f"Edge plane:\n{state[4]}")     # 應該顯示邊緣為 1
```

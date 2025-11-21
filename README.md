# AlphaZero-OmniFive

AlphaZero-OmniFive 將 AlphaZero 演算法套用在五子棋（Gomoku）上，完全依靠自對弈資料訓練策略價值網路，再結合蒙地卡羅樹搜索（MCTS）做決策。因為五子棋的狀態空間遠小於圍棋或西洋棋，只要一台具備 CUDA GPU 的PC，在幾個小時內就能獲得具有競爭力的棋力。

#### AlphaGo 與 AlphaGo Zero 的差異

- **AlphaGo**：結合專家棋譜、人工設計特徵與候選步預測，搭配 MCTS，並使用自我對局進一步強化。
- **AlphaGo Zero**：從零開始僅依賴規則自我對弈，使用殘差卷積網路同時輸出策略與價值，再搭配 MCTS；捨棄人工特徵與人類棋譜，架構更簡潔、訓練與推理更有效率，棋力也超越 AlphaGo。


## 系統需求

- 如果要遊玩需要下列套件
  - python >= 2.7
  - numpy >= 2.11
- 如果要訓練模型需要下列套件
  - Pytorch >= 0.4


## 初始設定

```bash
git clone https://github.com/Suyw-0123/AlphaZero-OmniFive.git
cd AlphaZero-OmniFive
```

### 使用 `config.cfg` 管理參數

專案根目錄的 `config.cfg` 採 INI 格式，分為三個主要區段：

| 區段 | 主要鍵值 | 說明 |
| --- | --- | --- |
| `[board]` | `width`, `height`, `n_in_row` | 棋盤大小與連珠數，訓練、對局共用 |
| `[training]` | `batch_size`, `n_playout`, `learn_rate` 等 | AlphaZero 自對弈與網路訓練的超參數 |
| `[human_play]` | `model_file`, `framework`, `start_player`, `n_playout` | 人機對戰時載入的模型與 MCTS 參數 |

任何調整只要修改 `config.cfg`，接著重新啟動 `train.py` 或 `human_play.py` 即可套用。

### 重新訓練不同棋盤

若想訓練 9x9 套路、11x11 或更大棋盤，只需在 `[board]` 區段更新尺寸與 `n_in_row`，接著重新執行 `train.py`。

## 訓練模型

```bash
python train.py
```

訓練流程包含：

1. 自對弈收集棋譜並做旋轉、翻轉增強。
2. 以小批次資料更新策略價值網路。
3. 固定回合對純 MCTS 對手評估，若勝率提升則覆寫 `best_policy.model`。

輸出模型：

- `current_policy.model`：最新訓練後的網路。
- `best_policy.model`：迄今評估中戰績最佳的網路。

## 與模型對局

```bash
python human_play.py
```

啟動後會顯示棋盤，輸入格式為 `row,col`（例如 `3,4`）。座標以左下角為 `(0,0)`，往上、往右遞增。若想調整對手棋力，可在 `config.cfg` 的 `[human_play]` 區段調整 `n_playout` 或改變 `framework`（PyTorch / NumPy）。

## 參考資料

- 重要感謝 https://github.com/junxiaosong/AlphaZero_Gomoku.git 提供核心代碼

- Silver et al., *Mastering the game of Go with deep neural networks and tree search* (Nature, 2016)
- Silver et al., *Mastering the game of Go without human knowledge* (Nature, 2017)

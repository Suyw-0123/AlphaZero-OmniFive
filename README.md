# AlphaZero-OmniFive

AlphaZero-OmniFive 將 AlphaZero 演算法套用在五子棋（Gomoku）上，完全依靠自對弈資料訓練策略價值網路，再結合蒙地卡羅樹搜索（MCTS）做決策。因為五子棋的狀態空間遠小於圍棋或西洋棋，只要一台具備 CUDA GPU 的PC，在幾個小時內就能獲得具有競爭力的棋力。

#### AlphaGo 與 AlphaGo Zero 的差異

- **AlphaGo**：結合專家棋譜、人工設計特徵與候選步預測，搭配 MCTS，並使用自我對局進一步強化。
- **AlphaGo Zero**：從零開始僅依賴規則自我對弈，使用殘差卷積網路同時輸出策略與價值，再搭配 MCTS；捨棄人工特徵與人類棋譜，架構更簡潔、訓練與推理更有效率，棋力也超越 AlphaGo。

![playout400](playout400.gif)

### 系統需求

- Python >= 3.9
- PyTorch >= 2.0（需支援 CUDA 的 GPU 驅動環境）
- numpy >= 1.24
- 建議安裝 [Ruff](https://docs.astral.sh/ruff/) 以配合推送前的靜態檢查（GitHub Actions 會在 push/pr 自動執行）。


### 初始設定

```bash
git clone https://github.com/Suyw-0123/AlphaZero-OmniFive.git
cd AlphaZero-OmniFive
```

## 與模型對局

```bash
python human_play.py
```


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

### 訓練參數說明

| 參數 | 預設值 | 說明 |
| --- | --- | --- |
| `board_width` / `board_height` | 8 | 棋盤長寬；8x8 為加強版五子棋常用設定。若調整需同時修改 `n_in_row`。 |
| `n_in_row` | 5 | 連成五子的勝利條件。與棋盤尺寸共同決定遊戲難度。 |
| `learn_rate` | 2e-3 | Adam 最初學習率。會依據 KL 發散程度由 `lr_multiplier` 動態縮放。 |
| `lr_multiplier` | 1.0 | KL 超過或低於門檻時成倍調整，控制學習率衰減或回升。 |
| `temp` | 1.0 | 自對弈時的溫度，決定落子探索度，後期可降溫以減少隨機性。 |
| `n_playout` | 900 | 每步行棋的 MCTS 模擬次數。數值越高越強但推理時間越長。 |
| `c_puct` | 5 | MCTS 探索係數，平衡高訪問率與高評分節點。 |
| `buffer_size` | 20000 | 自對弈資料緩衝區容量，越大代表保留更多歷史對局供訓練。 |
| `batch_size` | 768 | 每次梯度更新的樣本數。依 GPU 記憶體調整；8GB GPU 建議 640-768。 |
| `play_batch_size` | 2 | 單次自對弈產生的棋局數量，兩局可同時填充緩衝區。 |
| `epochs` | 8 | 每次更新時反覆疊代 mini-batch 的次數，提高收斂速度。 |
| `kl_targ` | 0.02 | KL 散度目標，限制新舊策略差距，配合 `lr_multiplier` 控制步幅。 |
| `check_freq` | 30 | 每蒐集多少批次自對弈資料就進行一次 MCTS 評估與模型儲存。 |
| `game_batch_num` | 2500 | 訓練迴圈上限；若提前 Ctrl+C 會保留目前的最佳模型。 |
| `pure_mcts_playout_num` | 2000 | 評估時純 MCTS 對手的模擬次數，數值越高評測越嚴格。 |

> 若遇到 GPU 記憶體不足，可先將 `batch_size` 調低到 512 或 384，並同步降低 `n_playout` 以縮短自對弈時間。

### 開發建議流程

1. `python -m ruff check .` 確認靜態檢查通過。
2. `python train.py` 或 `python human_play.py` 進行訓練 / 對弈驗證。
3. 提交前清理不必要的 `.model` 檔案；Git 版本庫已設定忽略。


## 參考資料

- 重要感謝 [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku.git) 提供核心代碼

- Silver et al., *Mastering the game of Go with deep neural networks and tree search* (Nature, 2016)
- Silver et al., *Mastering the game of Go without human knowledge* (Nature, 2017)

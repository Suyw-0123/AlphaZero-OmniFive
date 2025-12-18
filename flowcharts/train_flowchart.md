```mermaid
graph TD
    START([開始]) -->|命令行參數| PARSE["main()<br/>解析命令行參數<br/>--config, --init-model"]
    PARSE --> INIT["TrainPipeline.__init__()<br/>初始化訓練管道"]
    
    INIT --> LOAD_CFG["加載配置文件<br/>config.json<br/>(board, training, network)"]
    LOAD_CFG --> SETUP_BOARD["設定棋盤<br/>width, height, n_in_row"]
    SETUP_BOARD --> SETUP_PARAMS["載入訓練超參數<br/>learn_rate, batch_size<br/>epochs, kl_targ等"]
    SETUP_PARAMS --> CHECK_GPU{"GPU<br/>可用?"}
    
    CHECK_GPU -->|否| ERROR1["錯誤<br/>CUDA 不可用"]
    ERROR1 --> END1([中斷])
    
    CHECK_GPU -->|是| LOAD_MODEL["加載神經網路<br/>PolicyValueNet<br/>(num_channels=128<br/>num_res_blocks=4)"]
    LOAD_MODEL --> INIT_MCTS["初始化 MCTS 玩家<br/>MCTSPlayer<br/>(policy_value_fn)"]
    INIT_MCTS --> RUN["執行 run()"]
    
    RUN --> LOOP["迴圈: game_batch_num<br/>i = 0 to game_batch_num-1"]
    LOOP --> COLLECT["collect_selfplay_data()<br/>收集自我對弈資料"]
    
    COLLECT --> SELF_PLAY["<b>start_self_play()</b><br/>執行 play_batch_size 場遊戲<br/>每場遊戲:"]
    SELF_PLAY --> SP_DETAIL["├─ 初始化棋盤<br/>├─ 迴圈進行著法選擇:<br/>│  ├─ MCTS 搜索 n_playout 次<br/>│  ├─ 記錄狀態 state<br/>│  ├─ 記錄著法概率 mcts_prob<br/>│  └─ 執行著法<br/>└─ 遊戲結束<br/>└─ 返回 winner, play_data"]
    SP_DETAIL --> DATA_AUG["get_equi_data()<br/>資料增強"]
    
    DATA_AUG --> AUG_DETAIL["<b>旋轉 + 翻轉擴展</b><br/>├─ 對每局棋局<br/>├─ 旋轉 4 次 (0°,90°,180°,270°)<br/>├─ 翻轉 2 次 (水平/不翻)<br/>└─ 生成 8 倍數據<br/>└─ 存入 data_buffer"]
    AUG_DETAIL --> PRINT_LEN["打印: batch i, episode_len"]
    PRINT_LEN --> CHECK_BUFFER{"data_buffer<br/>size > batch_size?"}
    
    CHECK_BUFFER -->|否| SKIP_TRAIN["跳過網路訓練<br/>繼續收集資料"]
    CHECK_BUFFER -->|是| POLICY_UPD["policy_update()<br/>更新神經網路"]
    
    POLICY_UPD --> SAMPLE["隨機採樣小批次<br/>batch_size 個樣本"]
    SAMPLE --> GET_OLD["計算舊策略<br/>old_probs, old_v"]
    GET_OLD --> TRAIN_LOOP["訓練迴圈: epochs次<br/>i = 0 to epochs-1"]
    
    TRAIN_LOOP --> TRAIN_STEP["train_step()<br/>└─ 前向傳播<br/>└─ 損失函數:<br/>   L = L_policy + L_value<br/>   L_policy = -Σ π·log(p)<br/>   L_value = (z-v)²<br/>└─ 反向傳播 & 優化"]
    TRAIN_STEP --> GET_NEW["計算新策略<br/>new_probs, new_v"]
    GET_NEW --> CALC_KL["計算 KL 散度<br/>KL = Σ π_old · log(π_old/π_new)"]
    CALC_KL --> CHECK_KL{"KL > 4×<br/>kl_targ?"}
    
    CHECK_KL -->|是| BREAK["提前停止訓練<br/>避免過度更新"]
    CHECK_KL -->|否| NEXT_EPOCH["進入下一個 epoch"]
    BREAK --> NEXT_EPOCH
    NEXT_EPOCH --> EPOCHS_END{"已完成<br/>epochs?"}
    EPOCHS_END -->|否| TRAIN_LOOP
    EPOCHS_END -->|是| ADJUST_LR["動態調整學習率<br/>lr_multiplier"]
    
    ADJUST_LR --> CHECK_KL_HI{"KL > 2×<br/>kl_targ AND<br/>lr_multiplier > 0.1?"}
    CHECK_KL_HI -->|是| LR_DOWN["lr_multiplier /= 1.5<br/>降速學習"]
    CHECK_KL_HI -->|否| CHECK_KL_LO
    LR_DOWN --> CHECK_KL_LO{"KL < kl_targ/2<br/>AND<br/>lr_multiplier < 10?"}
    CHECK_KL_LO -->|是| LR_UP["lr_multiplier ×= 1.5<br/>加速學習"]
    CHECK_KL_LO -->|否| CALC_EXPL["計算可解釋方差<br/>explained_var_old/new"]
    LR_UP --> CALC_EXPL
    CALC_EXPL --> PRINT_STATS["打印統計信息<br/>kl, lr_multiplier,<br/>loss, entropy,<br/>explained_var"]
    PRINT_STATS --> TRAIN_END["返回 loss, entropy"]
    
    SKIP_TRAIN --> CHECK_EVAL
    TRAIN_END --> CHECK_EVAL{"(i+1) % check_freq<br/>== 0?<br/>定期評估"}
    
    CHECK_EVAL -->|否| NEXT_BATCH
    CHECK_EVAL -->|是| EVAL["policy_evaluate()<br/>新模型 vs 純MCTS"]
    
    EVAL --> EVAL_DETAIL["<b>評估遊戲</b><br/>├─ 執行 n_games 場對局<br/>├─ 新 MCTS player vs Pure MCTS<br/>├─ 計算勝率: win/(win+lose+tie)<br/>└─ 返回 win_ratio"]
    EVAL_DETAIL --> SAVE_CUR["保存當前模型<br/>→ current_policy.model"]
    SAVE_CUR --> CHECK_BEST{"win_ratio ><br/>best_win_ratio?"}
    
    CHECK_BEST -->|是| SAVE_BEST["新最佳模型!<br/>更新 best_win_ratio<br/>保存 → best_policy.model"]
    CHECK_BEST -->|否| CHECK_PLAYOUT
    SAVE_BEST --> CHECK_PLAYOUT{"best_win_ratio<br/>== 1.0 AND<br/>pure_mcts_playout<br/>< 5000?"}
    
    CHECK_PLAYOUT -->|是| UP_PLAYOUT["提高難度<br/>pure_mcts_playout_num += 1000<br/>重置 best_win_ratio = 0.0"]
    CHECK_PLAYOUT -->|否| NEXT_BATCH
    UP_PLAYOUT --> NEXT_BATCH
    
    NEXT_BATCH{"完成所有<br/>game_batch_num<br/>批次?"}
    
    NEXT_BATCH -->|否| LOOP
    NEXT_BATCH -->|是| SUCCESS["訓練完成!<br/>best_policy.model<br/>已保存"]
    SUCCESS --> NORMAL_END([正常結束])
    
    TRAIN_LOOP -.->|KeyboardInterrupt| INTERRUPT["用户中斷<br/>Ctrl+C"]
    INTERRUPT --> QUIT["quit"]
    QUIT --> QUIT_END([異常結束])
    
    style START fill:#90EE90
    style NORMAL_END fill:#FFB6C6
    style QUIT_END fill:#FF6B6B
    style ERROR1 fill:#FF6B6B
    style BREAK fill:#FFE4B5
    style SAVE_BEST fill:#FFD700
    style UP_PLAYOUT fill:#87CEEB
    style SP_DETAIL fill:#E6E6FA
    style AUG_DETAIL fill:#E6E6FA
    style EVAL_DETAIL fill:#E6E6FA
```

## 流程圖詳細說明

### 🔴 主要階段

####  **初始化階段** (START → RUN)
```
├─ 解析命令行參數
├─ 加載 config.json
├─ 設定棋盤參數
├─ 檢查 GPU 可用性
├─ 加載神經網路 (ResNet)
└─ 初始化 MCTS 玩家
```

####  **自我對弈資料收集** (collect_selfplay_data)
```
play_batch_size 場遊戲:
├─ 初始化棋盤
├─ 循環執行著法 (直到遊戲結束):
│  ├─ MCTS: n_playout 次模擬搜索
│  ├─ 著法選擇: temperature=temp
│  ├─ 記錄: (state, mcts_prob)
│  └─ 執行著法
├─ 返回: (winner, play_data)
└─ 儲存 episode_len
```

####  **資料增強** (get_equi_data)
```
對每局棋:
├─ 旋轉 4 次 (0°, 90°, 180°, 270°)
└─ 對每個旋轉做翻轉 (2 次)
   └─ 生成 8 倍等價棋局
```

####  **網路訓練** (policy_update)
```
if data_buffer.size > batch_size:
  ├─ 隨機採樣 mini_batch
  ├─ 計算舊策略 (old_probs, old_v)
  ├─ 訓練迴圈 (epochs 次):
  │  ├─ 前向傳播
  │  ├─ 計算損失: L = L_policy + L_value
  │  ├─ 反向傳播 + 優化
  │  ├─ 計算新策略 (new_probs, new_v)
  │  ├─ 計算 KL 散度
  │  └─ if KL > 4×target: break (提前停止)
  │
  ├─ 動態學習率調整:
  │  ├─ if KL > 2×target: lr_multiplier /= 1.5 (降速)
  │  └─ if KL < target/2: lr_multiplier ×= 1.5 (加速)
  │
  └─ 打印統計信息
```

####  **定期評估** (policy_evaluate)
```
if (i+1) % check_freq == 0:
  ├─ 執行 n_games 場評估遊戲
  ├─ 對手: Pure MCTS (N=pure_mcts_playout_num)
  ├─ 計算勝率
  ├─ 保存 current_policy.model
  │
  └─ if win_ratio > best_win_ratio:
     ├─ 保存 best_policy.model (新最佳)
     └─ if win_ratio == 100%:
        └─ 提高難度: pure_mcts_playout_num += 1000
```

###  關鍵變數追蹤

|            變數             |                 用途              |
|-----------------------------|-----------------------------------|
| **data_buffer**             | 儲存自我對弈資料 (最多 buffer_size) |
| **lr_multiplier**           | 動態學習率倍數 (範圍: 0.1~10)       |
| **best_win_ratio**          | 追蹤最佳模型性能                    |
| **episode_len**             | 每場遊戲的步數                      |
| **pure_mcts_playout_num**   | 評估難度 (逐漸增加)                 |

###  性能最佳化點

1. **早期停止 (Early Stopping)**
   ```python
   if kl > self.kl_targ * 4:
       break  # 避免過度訓練
   ```

2. **自適應學習率**
   ```python
   if kl > 2×target:
       lr_multiplier /= 1.5  # 太快降速
   elif kl < target/2:
       lr_multiplier *= 1.5  # 太慢加速
   ```

3. **漸進式難度提升**
   ```python
   if win_ratio == 1.0 and playout < 5000:
       pure_mcts_playout_num += 1000  # 邪惡難度
   ```

###  決策邏輯

```
訓練流程決策樹:

START
  ↓
批次 i (0 to game_batch_num-1)
  ├─ 收集 play_batch_size 場自我對弈
  ├─ 資料增強 (8×擴展)
  ├─ buffer 足夠? → 訓練網路
  │  └─ KL 監控 & 學習率調整
  │
  └─ 每 check_freq 批:
     ├─ 評估 vs Pure MCTS
     ├─ 超越最佳? → 保存新最佳
     └─ 100% 勝率? → 提高 Pure MCTS 難度(self play MCTS模擬次數 +1000)
       └─ 重置評估 (繼續進步)
```

###  中斷處理

```
try:
  執行訓練迴圈
except KeyboardInterrupt:
  print('quit')  # (Ctrl+C)
```

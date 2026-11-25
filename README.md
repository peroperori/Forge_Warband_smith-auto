# VRChat《爐火戰團》鐵匠小遊戲自動工具

適用於 VRChat 桌面視窗模式。工具會偵測紅色判定條進入金色區域時，自動在原地送出滑鼠左鍵點擊。

## 環境需求

- Windows 10 / Windows 11（64 位元）
- Python 3.10 或更新版本
- VRChat 桌面模式視窗

## 安裝

建議使用 `uv` 建立虛擬環境：

```powershell
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[all]"
```

或使用標準 `pip`：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## 快速開始

1. 啟動 VRChat 並進入桌面模式的鐵匠小遊戲。
2. 先執行乾跑模式與即時預覽，確認偵測正常：

```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto run --dry-run --preview
```

此模式不會移動游標、點擊滑鼠或按下 Tab。在預覽視窗按 `q` 或關閉視窗即可退出。

3. 確認偵測正常後執行：

```powershell
.\.venv\Scripts\vrc-smith-auto.exe run
```

4. 將滑鼠游標停在畫面上的點擊區域（例如「锤!」按鈕上方）。工具偵測到金色區域且 VRChat 為前景視窗時，會在原地點擊。

## 熱鍵

| 熱鍵 | 功能 |
|---|---|
| `F8` | 暫停或恢復自動點擊 |
| `Esc` | 緊急停止並釋放滑鼠左鍵 |

## 指令

以下指令皆可使用模組方式執行：

```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto <指令> [參數]
```

也可使用已安裝的執行檔：

```powershell
.\.venv\Scripts\vrc-smith-auto.exe <指令> [參數]
```

### 即時執行

```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto run [--config PATH] [--backend auto|dxcam|mss|gdi] [--dry-run] [--preview] [--no-focus-guard]
```

- `--config PATH`：載入自訂 YAML 或 JSON 設定檔。
- `--backend`：選擇螢幕擷取方式，預設為 `auto`。
- `--dry-run`：只偵測，不送出滑鼠點擊。
- `--preview`：開啟即時偵測預覽視窗。
- `--no-focus-guard`：允許 VRChat 非前景時執行，通常僅供離線測試。

### 離線驗證

```powershell
# 單張截圖：確認非金色區不會誤觸
.\.venv\Scripts\python.exe -m vrc_smith_auto verify-image --image PATH --expect-clicks 0 --json

# 整段影片：以影片時間戳逐帧重放
.\.venv\Scripts\python.exe -m vrc_smith_auto verify-video --video PATH --dry-run --json

# 單帧延遲基準（p95 門檻預設 16.7ms）
.\.venv\Scripts\python.exe -m vrc_smith_auto benchmark --video PATH --iterations 200 --max-p95-ms 16.7 --json
```

## 高速末段調校

末段金色區會縮到約 `18px`、红條速度可達約 `500px/s`，單帧就移動半個判定區。本作是**滑鼠放開才判定**，所以 `Runner` 會依 `release_delay_s` 把 `LEFTUP` 排程到「預測红條在中心」的時刻；`lead_ms` 代表**從擷取到的帧到遊戲實際判定**的總延遲（含偵測約 10ms 與遊戲下一次輸入輪詢約 16.7ms）。若仍然「只有最高難度才失手」：

- `trigger.lead_ms`：預設 `8.3` 是保守值（避免红條在金色外折返時被預測成進入）。最高難度建議先試 `16.7`，再試 `25`–`33.3`；若反而打到銀/灰（提前），往下調。
- `trigger.max_lead_offset`：預測位移上限（norm 單位）。預設 `2.0`；太小會補不到末段。
- `trigger.hold_ms`：`1.0` 是最小按住時間；實際放開可能再延後最多約 30ms 以對準中心。若點擊偶爾完全沒反應（遊戲沒收到 down+up），可試 `3`–`5`；每多 1ms 在末段只多約 0.5px。
- 失敗方向判斷：红條已過金色才判定 → 提高 `lead_ms`；红條還沒到中心就判定（銀/灰）→ 降低 `lead_ms`；完全沒判定 → 提高 `hold_ms`。
- 執行時不要開 `--preview`，並確認主迴圈能穩定 60fps（單帧偵測 p95 < 16.7ms）。

## 設定檔

建立 `config.yaml`，再透過 `--config` 載入：

```yaml
capture:
  backend: auto
  target_fps: 60
  monitor_idx: 0
  crop_to_window: true

window:
  title: VRChat
  class_name: UnityWndClass
  require_foreground: true
  rect_refresh_s: 1.0

trigger:
  hold_ms: 1.0
  cooldown_ms: 200.0
  entry_margin: 0.95
  exit_margin: 1.25
  lead_ms: 8.3
  max_lead_offset: 2.0
  rearm_lost_frames: 3
  dry_run: false
  auto_fire_up: false
  auto_sell: false
```

- `auto_fire_up: true`：辨識到「锤!」停用時，自動點擊「开炉」。等「锤!」恢復可用後，將游標移到按鈕並恢復金色區域判定。
- `auto_sell: true`：搭配 `auto_fire_up` 使用。點擊「开炉」後從 500 毫秒開始檢查按鈕狀態，至少等待 1.5 秒，並在連續兩次辨識到可用的「开炉」與停用的「锤!」時按 Tab 開啟選單。進入「铁匠铺」逐格雙擊出售倉庫物品。確認倉庫清空後，再按 Tab 並恢復鍛造。此設定會出售倉庫內所有物品。
- 按鈕與倉庫的辨識圖片以 1920×1080 為基準，會依 VRChat 用戶端視窗的大小調整。請讓完整的遊戲視窗留在擷取範圍內。

載入方式：

```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto run --config config.yaml
```

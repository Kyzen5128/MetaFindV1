# MetaFind 復現

復現 *MetaFind: Scene-Aware 3D Asset Retrieval for Coherent Metaverse Scene Generation*
（論文全文在 [`docs/metafind_paper.md`](docs/metafind_paper.md)）。

單張 RTX 4090、frozen ULIP-2 backbone、本地 Qwen 取代 GPT-4o。

## 快速開始

```bash
bash setup/01_storage.sh          # 建 /mnt/data1/kyzen/MetaFind 並做 ./data symlink（需 sudo）
bash setup/02_conda_env.sh        # 建 conda 環境 MetaFind
conda activate MetaFind
python setup/03_verify_env.py     # 驗證環境（加 --full 會下載 10GB 的 ViT-bigG-14）
python -m pytest tests/ -q        # 單元測試
```

## 目錄結構

```
metafind/            我們寫的程式
  models/            ESSGNN、模態融合、雙塔對比 loss、雙塔模型、ULIP-2 backbone 封裝
  data/              資料抓取、完整性驗證、ProcTHOR 場景圖
  compat/            ULIP 在現代 PyTorch 上的 runtime 修補（不改上游原始碼）
  vendor/            上游第三方原始碼（ULIP、EGNN）→ 見 vendor/README.md
setup/               環境建置與驗證
tests/               單元測試
docs/
  metafind_paper.md  論文
  graph/             設計文件 → 見 graph/README.md
data ->              symlink 到 /mnt/data1/kyzen/MetaFind（大型資料，不進 git）
scratch/             參考用的雜項腳本
```

## 設計文件

`docs/graph/` 是用 graph-engineering 方法產出的完整規格：

| 檔案 | 內容 |
|---|---|
| [`00_FINDINGS.md`](docs/graph/00_FINDINGS.md) | 實際檢查論文與程式碼後的硬事實，**包含論文的多處自相矛盾** |
| [`01_GRAPH_SPEC.md`](docs/graph/01_GRAPH_SPEC.md) | 節點、state、邊、路由、迴圈、失敗政策、gate、可觀測性 |
| [`02_BUILD_STEPS.md`](docs/graph/02_BUILD_STEPS.md) | 逐步驟做什麼、每步的通過條件 |
| `*.yaml` | 結構化規格（可程式化檢查） |

## 已知偏離論文之處

| id | 內容 |
|---|---|
| **D1** | 用官方釋出的 ULIP-2 checkpoint，不自行預訓練（官方腳本假設 8 張 GPU） |
| **D2** | frozen backbone 的輸出預先算好快取，訓練只在 1280-d 向量上進行 |
| **F1** | 論文 §2.5 的 `h⁰=Concat(x,t)` 與 Appendix C 的等變性證明前提矛盾 |
| **F10** | 論文 §2.5 的 `f_x → ℝ³` 與證明矛盾（必須是純量，否則旋轉提不出來） |
| **F11** | `e_layout` 只讀 `h`，最後一層的座標 MLP 收不到梯度 |
| **F12** | ProcTHOR 實測 1,467 個 unique asset，論文說「3,000+」 |

完整清單與證據在 [`docs/graph/00_FINDINGS.md`](docs/graph/00_FINDINGS.md)。

## 授權

MetaFind 復現程式碼見本 repo。`metafind/vendor/` 下的第三方程式碼各自沿用原授權
（ULIP: BSD-3-Clause，EGNN: MIT）。

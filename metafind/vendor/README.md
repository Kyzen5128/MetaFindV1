# `metafind/vendor/` — 上游第三方原始碼

MetaFind 是本專案要復現的東西，程式碼在 `metafind/models/`、`metafind/data/`、
`metafind/compat/`。這個資料夾放的是 MetaFind **依賴**的別人的程式，
只複製**實際用到的檔案**，不是整個 repo 搬過來。

## 為什麼要放進 repo

原本是寫死 `/home/kyzen/ULIP` 這種絕對路徑。那樣 repo 不是自足的 ——
換機器或那些 clone 被刪掉就跑不起來，而且沒有紀錄當初用的是哪個版本。

## 內容

| 路徑 | 來源 | 授權 | 檔案數 |
|---|---|---|---|
| `ulip/` | [salesforce/ULIP](https://github.com/salesforce/ULIP) | BSD-3-Clause | **19** |
| `egnn_clean.py` | [vgsatorras/egnn](https://github.com/vgsatorras/egnn) @ `e9ca6c0` 的 `models/egnn_clean/egnn_clean.py` | MIT（`LICENSE.egnn`） | **1** |

### ULIP 為什麼是這 19 個

用實際 import 追蹤（不是靜態分析）量出 `ULIP2_PointBERT_Colored` 建構 +
`encode_pc` / `encode_text` 會載入哪些檔案，只留這些：

```
models/ULIP_models.py            models/losses.py
models/pointbert/{point_encoder,dvae,misc,checkpoint,logger}.py
models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml
models/pointnet2/{pointnet2,pointnet2_utils}.py
data/dataset_3d.py
utils/{__init__,build,config,io,logger,registry,utils}.py
LICENSE.txt
```

`pointnet2` 我們沒用到 PointNet++，但 `ULIP_models.py` 在模組層級就
`from models.pointnet2.pointnet2 import Pointnet2_Ssg`，不留著連 import 都會失敗。

**沒有搬的**：PointNeXt、PointMLP、`main.py`、`scripts/`、`assets/`（101 MB 的 GIF）、
以及各種 CONTRIBUTING / CODE_OF_CONDUCT 等文件。從 450 個檔案（8.1 MB）縮到 19 個（200 KB）。

### EGNN 為什麼只有一個檔案

`egnn_clean.py` 只依賴 torch，是唯一被 import 的。整個 egnn repo 的其他部分
（qm9、n-body、autoencoder）跟 MetaFind 無關。

完整性用**內容雜湊**釘住（`UPSTREAM_SHA256`），而不是留第二份副本來 diff ——
留副本只是為了比對的話，那份副本本身就是冗餘。

### 順帶解掉的一個坑

ULIP 和 egnn **都有頂層 `models` 套件**。ULIP 的沒有 `__init__.py`（namespace package），
egnn 的有（regular package）—— Python 一律優先採用後者，**而且與 `sys.path` 順序無關**。
所以只要 egnn 被 import 過，ULIP 的 `models.pointbert` 就永遠拿不到。
把 egnn 抽成單檔模組 `metafind.vendor.egnn_clean` 之後，`models` 就只屬於 ULIP。

## 修改原則

**不要改這裡的檔案。** ULIP 在現代 PyTorch 上的相容性問題（`torch._six` 已移除、
兩個 CUDA extension 裝不起來、config 寫死相對路徑）全部由
`metafind/compat/ulip_patch.py` 在 runtime 修補。這樣上游可以隨時重新同步，
而我們的修補是一份可讀的清單，不會混在別人的程式碼裡。

# `metafind/vendor/` — 上游第三方原始碼

這個資料夾放的是**別人寫的程式碼**，我們原封不動搬進來（vendoring）。
MetaFind 自己的程式碼在 `metafind/models/`、`metafind/data/`、`metafind/compat/`。

## 為什麼要搬進來而不是用外部路徑

原本這些程式碼放在 `/home/kyzen/ULIP` 和 `/home/kyzen/egnn`，程式裡寫死絕對路徑。
那樣的話 repo **不是自足的** —— 換一台機器、或那兩個 clone 被刪掉，整個模型就跑不起來，
而且沒有任何紀錄說明當初用的是哪個版本。

搬進來之後：clone 這個 repo 就有完整的模型，不需要任何外部前置。

## 內容

| 路徑 | 來源 | 授權 | 我們怎麼用 |
|---|---|---|---|
| `ulip/` | [salesforce/ULIP](https://github.com/salesforce/ULIP) | BSD-3-Clause（`ulip/LICENSE.txt`） | ULIP-2 backbone。`metafind/models/ulip_backbone.py` 把這個目錄加到 `sys.path`，然後 `from models.ULIP_models import ULIP2_PointBERT_Colored` |
| `egnn/` | [vgsatorras/egnn](https://github.com/vgsatorras/egnn) commit `e9ca6c0` | MIT（`egnn/LICENSE`） | 只作為出處存查，程式不直接 import 它 |
| `egnn_clean.py` | 上面那份的 `models/egnn_clean/egnn_clean.py` | MIT（`LICENSE.egnn`） | **實際被 import 的檔案**。`metafind/models/essgnn.py` 從這裡取 `unsorted_segment_sum` 等函式 |

## 為什麼 EGNN 要複製兩份

因為 **套件名衝突**。

ULIP 和 egnn **都有一個叫 `models` 的頂層套件**。ULIP 的沒有 `__init__.py`（namespace
package），egnn 的有（regular package）—— Python 一律優先採用後者，**而且跟 `sys.path`
順序無關**。所以只要 egnn 被 import 過一次，ULIP 的 `models.pointbert` 就永遠拿不到。

MetaFind 同時需要兩邊，所以這是結構性衝突，不是設定問題。

解法是把 egnn 唯一用到的那個檔案抽出來變成 `metafind.vendor.egnn_clean`
（它只依賴 torch，沒有其他相依），這樣 `models` 就只屬於 ULIP。
`egnn/` 全套仍然留著，用途是存查與 diff。

`tests/test_essgnn.py` 有一條測試會比對 `egnn_clean.py` 與 `egnn/` 裡的原始檔
**逐位元組相同**，避免哪天有人偷偷改了抽出來的那份而沒人發現。

## 修改原則

**不要改這裡的檔案。** ULIP 在現代 PyTorch 上有相容性問題（`torch._six` 已移除、
兩個 CUDA extension 裝不起來、config 路徑寫死相對路徑），全部由
`metafind/compat/ulip_patch.py` 在 runtime 修補，而不是改動上游程式碼。

這樣做的好處是：上游版本可以隨時重新同步，而我們的修補是一份可讀的清單，
不會混在幾千行別人的程式碼裡。

## 排除的內容

`ulip/assets/`（101 MB 的 GIF 動畫）沒有搬，對模型無用。
各 vendored repo 自帶的 `.gitignore` 也移除了 —— 它們是上游的 repo 衛生規則，
留著會**靜默排除**我們想追蹤的 vendored 檔案（egnn 那份就擋掉了自己的 `models/egnn.png`）。

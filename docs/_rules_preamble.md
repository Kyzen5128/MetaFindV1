## 產生本快照時發現的規則衝突（**需 Kyzen 裁決**）

### 衝突 C-R1：上游值到底能不能採用

**規則 2（論文復現）§3 Paper Silence** 明文禁止：

> Do not infer a value merely because:
> - **an upstream repository uses it**
> - a library defaults to it

**但 Kyzen 的 standing rule**（2026-08-25 口頭、2026-08-26 再次下令並要求寫入規則）是：

> 「若 metafind論文找不到答案去找上游的論文」
> 「找不到答案高機率是照原架構的方法」

**這兩條在決策方向上相反。** 規則 2 說「上游用了不構成採用理由」，standing rule 說「論文沒講就照上游」。

**影響範圍不小。** 本輪依 standing rule 採用或建議採用的值，全部落在規則 2 §3 的禁止清單裡：

| 值 | 來源 | 規則 2 §3 的字面判定 |
|---|---|---|
| Stage 1 epochs 250 | ULIP `main.py:47` default ＋ 官方腳本 | 「a library defaults to it」→ 禁止 |
| 不早停、取 best checkpoint | ULIP `main.py:212-231`、OpenShape `train.py:190-201` | 「an upstream repository uses it」→ 禁止 |
| ESSGNN 層數 7 | EGNN `main_qm9.py:34` | 同上 |
| ESSGNN pooling sum | EGNN `qm9/models.py:83` | 同上 |
| lr 建議 5e-4 | OpenShape supp:190 | 這條是**論文**不是 repo，規則 2 §3 不禁 |

**可能的調和讀法**（我的判讀，非裁決）：規則 2 §3 第一句其實留了門——
"classify it as UNKNOWN **unless another authoritative source resolves it**"，
而規則 2 §4 又說上游細節在「有證據 MetaFind 採用／繼承／依賴」時即為 MetaFind-relevant。
MetaFind 明文建構在 ULIP-2 與 EGNN 之上，所以上游**可以**算 authoritative source。
若採此讀法，§3 真正禁止的是「照抄後當成已解決、標成 PAPER FACT」，
而不是「採用上游值並標成 UPSTREAM FACT」。

**我目前的做法一律標 UPSTREAM FACT ＋ 附檔案行號，從未標成 PAPER FACT。**
但字面衝突仍在，需要 Kyzen 用一句話定調，二選一：

- **甲**：規則 2 §3 加註「上游官方論文與程式碼算 authoritative source；採用時標 UPSTREAM FACT，不得標 PAPER FACT」。standing rule 勝出。
- **乙**：standing rule 只適用於「架構與方法」，不適用於「超參數數值」；數值一律 UNKNOWN 並上呈 Kyzen。規則 2 §3 勝出。

**在 Kyzen 定調前，本輪所有依 standing rule 採用的值都維持「建議」狀態，不寫入協定。**

### 衝突 C-R2（較輕）：什麼時候該停下來問人

**規則 1（研究嚴謹度）§2** 列出一長串「會影響研究結果就 STOP 並詢問使用者」的清單，
其中包含 `hyperparameters`、`training procedure`、`optimization`。
**規則 5（上游查找）** 則要求四步查完才准上呈。

這兩條**不矛盾、是順序關係**（先查完再問），但規則 1 §2 字面沒有「先查上游」這一步。
建議在規則 1 §2 加一句指向規則 5，避免下一個讀到的人又直接跳去問人。
此項不影響現有結論，屬文件整併。

---

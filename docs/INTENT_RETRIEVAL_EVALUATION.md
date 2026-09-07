# 家具需求檢索評估

主評估回答的是：**使用者提出家具需求後，模型能否從固定物件庫找到至少一個可接受的選項？**
同一需求可以有多個正解。找到另一件同樣合適的椅子，也可以成功，不要求找回參考圖片所屬的唯一資產。

這是使用者於 **2026-09-08 批准的本地 IMPLEMENTATION CHOICE**，不是作者 Table 1 未公開協定的還原。
真實需求題庫與逐條件人工正解仍待建立；本評估尚未產生正式研究結果。示意資料、程式可執行或舊同資產診斷，都不能替代這兩項工作。

## 一道需求可以有多個答案

例如：「找一張適合小餐桌、木質外觀的餐椅。」
使用者要的是符合需求的家具；物件庫裡可能有幾張不同椅子都合適。

- `query_id` 是需求編號，例如 `need-001`，與候選資產的 `asset UID` 分開。需求不必對應候選庫裡的某一件原始資產。
- 文字描述使用者的需求；圖片或點雲是可選的參考線索。沒有提供的線索保持缺席，不從某個正解資產自動補入。
- 每題、每個可見模態條件，都有自己的 relevance labels，以下稱 `qrels`：候選可以接受記為 `1`，不能接受記為 `0`。
- 圖片條件只能依圖片可見的線索審核。例如文字裡的尺寸限制沒有提供給 I-only 模型時，不可以偷偷用這個限制判錯。

這不是把類別相同的物件全部判成正解。是否接受，要依事先聲明的需求、可見線索和審核依據決定。

## 主評估比較什麼

| 模型列 | 查詢端 | 候選端 |
|---|---|---|
| `ulip2_available_mean` | 官方預訓練 ULIP-2；可用模態先各自 L2 normalize，平均後再 normalize | 同一預訓練模型的完整 T/I/P 融合 |
| `stage1` | 還原指定 Stage 1 的實際 point encoder、fusion 與缺模態處理 | 該 Stage 1 的 gallery encoder |
| `stage2_layout_off` | 還原同一父模型上的 Stage 2 查詢 fusion，關閉 layout | 與 `stage1` 完全相同的候選向量 |

候選庫的資產清單、來源與順序先固定，每件都提供完整的文字、圖片和點雲，再預先編碼。
它不會隨當前 query 的可見模態改成只有 T、只有 I 或只有 P。
不同模型可以有不同候選向量；Stage 2-off 與指定父 Stage 1 的候選向量必須相同。

Stage 1 相對 mean 的差異可能包含 point encoder 訓練，不能全部歸因於融合方法。
Stage 2-off 的結果只回答物件檢索問題，不證明 ESSGNN 的場景資訊有用，也不代表場景品質或 Table 2 成績。
沒有提供 Stage 2 record 時只能比較前兩列，不能把缺少的第三列補成零分。

### 模態條件與分母

| CLI／qrels 條件名稱 | 實際提供的線索 |
|---|---|
| `text` | T：需求文字 |
| `image` | I：參考照片 |
| `pc` | P：參考點雲 |
| `text+image` | T+I |
| `text+pc` | T+P |
| `image+pc` | I+P |
| `full` | T+I+P |

七種條件只在參考輸入足夠時安排。每個條件的固定 query 清單與題數都要記錄；同一條件下，各模型使用完全相同的題目與答案。
若不同條件使用不同題池，不能把分數差直接解讀為新增模態的收益。要比較 T 與 T+I 的增益，應事先固定兩者共同可評的題目，並各自完成相應 qrels。

**新主評估不承諾固定 42 組。** 原本「兩觀測 × 三模型 × 七模態」的 42 組仍是獨立的同資產辨識診斷，見 [舊自訂評估](CUSTOM_TABLE1_EVALUATION.md)。

## 先固定題目與逐條件正解

1. **先選候選池。** 固定完整 UID 清單；v1 要求對這個候選池逐件審核。可以先選較小的固定池，但不能看模型分數後縮小候選池。
2. **再寫需求與整理參考。** 記錄文字、原始照片／點雲的來源、建立者與方法。不要從 canonical gallery UID 自動產生唯一正解。
3. **依條件審核。** 審核者只能使用該條件向模型提供的線索；每個 query-condition 的 qrels 必須覆蓋候選池全部 UID，每個值是 `0/1` 或布林值，至少有一個正例。
4. **先封存、後排名。** 保存 assessor、method、criteria，將輸入、qrels、來源身分和版本凍結，再產生模型輸出。不能根據模型排名補選正解或改寫判準。

未審完的候選池是 **待完成**。未標註不等於不相關，不能默認為 `0`；這種資料不得產生完整主評估分數。
若完整審核後某題沒有任何正例，該題不符合本版「至少一個可接受候選」的前提，須在模型作答前處理題目／候選範圍並記錄新版本，不能偷偷刪題或將 Recall 的零分母當成零分。

Gemma 標註的尺寸、重量和材質是模型估計，**不能直接當作實測 ground truth**。
審核依據可以使用已聲明的可核對來源與人工觀察；若某個硬性尺寸要求無法證實，就必須在題目、判準或資料狀態中如實處理，不能讓估計值冒充量測。
本文件不自動指定新的裁判模型、評分 prompt 或人工人數；這些不是執行程式應自行補出的設定。

### ULIP-2 原始描述可以怎麼用

使用者提供的原始資料位於：

```text
/mnt/data1/kyzen/datasets/ulip2_objaverse_lvis/ULIP-2/objaverse_lvis
```

原始 shard 的已查看樣本包含 `text`、`blip_caption`、`msft_caption`、`retrieval_text`，可以作為題目草稿或外部描述來源。
它們仍可能是機器生成的資產描述，不會自動變成真實使用者需求；採用前須審核內容是否適合該需求，保留 shard／成員／欄位身分，並另外建立逐模態條件的多正解 qrels。

既有抽取檔：

```text
/home/kyzen/metafind/metafind_data/outputs/_probe/ulip2_query_feats/ulip2_query_texts.json
```

這是**舊 corpus 的快取**，不是已核對新 corpus 交集的清單。不能據此宣稱新候選池的描述均已齊全。
原始 shard 已查看樣本只有圖片特徵，沒有可直接作參考照片的 raw image；需要另外取得與記錄實際圖片來源。
原始 xyz/rgb 點資料則可在記錄來源後整理成輸入格式，再由本次模型重新編碼。

不要因為舊 `text_feat`、`thumbnail_feat` 或 `image_feat` 的維度相同，就把它們當成本次相同權重產生的向量。
本流程接收原始文字、圖片與點雲，不接收來源未對齊的外部 embedding。
另用官方 caption 查回同一個 UID 可以作診斷，但仍不是這裡的多答案需求評估。

抽取工具 [prepare_ulip2_intent_queries.py](../tools/prepare_ulip2_intent_queries.py) 只讀取指定 UID 所屬的 shard，產生待審核草稿，不產生需求正解。
先準備非空、無重複的 Objaverse UID JSON 列表，例如 `/path/to/source_asset_uids.json`；這是**草稿來源清單**，不會自動成為候選池或正例清單。
`--metadata` 必須是含 `entries` 列表的 JSON；指定 UID 的 entry 需有 `u`，以及精確的 `glb: "glbs/000-xxx/<UID>.glb"` 對應，讓工具找到正確 shard。

```bash
cd /home/kyzen/MetaFindV1
PY=/home/kyzen/miniconda3/envs/MetaFind/bin/python
$PY tools/prepare_ulip2_intent_queries.py \
  --shards /mnt/data1/kyzen/datasets/ulip2_objaverse_lvis/ULIP-2/objaverse_lvis \
  --asset-uids /path/to/source_asset_uids.json \
  --metadata /home/kyzen/upstream/openshape-objaverse-embeddings/objaverse_meta.json \
  --caption-field msft_caption \
  --out-dir /path/to/ulip2_need_drafts_run01
```

可選欄位是 `blip_caption`、`msft_caption`、`text`。工具保留指定欄位的原文，不在缺值時改用另一欄；`text` 若是列表，只接受恰好一個字串，不自行拼接。
需要原始點資料時另加 `--include-pointcloud`；工具只合併 xyz/rgb 並轉為 float32，不重新採樣、不改座標，正式 prepare 再驗證 `(10000, 6)` 與可用空間範圍。
圖片特徵不會被轉成照片，因此此抽取工具不提供 I 條件。

輸出 `draft_queries.json` 的狀態是 `needs_relevance_review`，包含來源 UID、shard／成員與 caption 身分，**沒有 `qrels` 或 `judgments`，不能直接交給正式 prepare**。
須人工確認需求、整理到下方 spec 的 `queries` 結構，再補齊逐條件全候選審核。不要直接複製整個草稿 record：正式 query 不接受 `id`、`status` 或 `source_uid` 額外欄位；來源 UID 可移入 `provenance` 供追溯，不會自動判成正例。
缺資料／空欄位會留在 `failures` 且命令回傳非零狀態，不得把部分草稿當成指定來源都已齊全。為檢查重複成員，即使只指定一個 UID，也會掃描其完整 shard；未使用的 shard 不讀取。工具綁定抽取成員的 SHA256 與 archive 路徑／stat；不宣稱已對整個大型 archive 做完整內容 hash。

## 怎麼計分

令 `P(q,c)` 為題目 `q` 在模態條件 `c` 下全部已審核的正例，`Top_k(q,c)` 為模型前 k 個候選：

```text
Hit@k(q,c)    = 1，若 Top_k 與 P 至少有一個交集；否則為 0。
Recall@k(q,c) = |Top_k ∩ P| / |P|。
```

主報告使用 **Hit@1、Hit@5**；輔助報告使用真正的 **Recall@1、Recall@5**。
每個模態條件各自對固定題目取平均，再乘 100%：每題權重相同，不按正例數加權。
結果同時列出該條件的 query 數、候選數，以及各題正例數，不能只留一個百分比。

例如一個**純教學示意**候選池有 8 件物件，其中 3 件可接受，排名分別為 2、6、8：

| 指標 | 這一道示意題的結果 |
|---|---|
| Hit@1 | 0%，第一名不合適 |
| Hit@5 | 100%，前五名已有一個合適答案 |
| Recall@5 | 1/3 = 33.3%，只找回三個答案中的一個 |

這些是說明公式的假設排名，**不是已執行的模型分數或真實 qrels**。

排序採 float64 cosine，由大到小；分數相同時按 gallery UID 字典序由小到大，不用正解優先打破同分。
候選數小於 k 時，前 k 名只包含現有候選；小候選池可能很容易達到 Hit@5，因此必須報告候選數，也不能直接拿去比較論文的大池分數。

## 與論文和舊評估的差別

| 項目 | 論文或既有診斷 | 本評估 |
|---|---|---|
| 查詢／答案 | 作者完整 query 配對與 relevance 規則未公開 | 真實家具需求，依每個可見模態條件標註多個可接受候選 |
| 指標 | 論文稱 top-k retrieval accuracy 為 R@1／R@5 | 主指標明稱 Hit@1／Hit@5，另報真正 Recall；不聲稱與作者指標完整等價 |
| 模態 | 論文列出七種組合 | 有足夠參考輸入才安排；逐條件固定分母 |
| P-only baseline | 論文指出其他模型 query／gallery 使用相同 PC embedding，導致高分 | 本地 mean 的 query=P，gallery 固定完整 T/I/P 融合；兩邊並非同一向量 |
| 候選池 | 論文稱 Objaverse-LVIS 約 48K；確切 gallery 名單未知 | 公開本次固定候選清單、數量與來源；完整審核這個範圍 |
| 舊 42 組 | 兩觀測下是否找回同一 UID | 保留為另外一份辨識診斷，不是主需求評估 |
| 訓練與選模 | 作者與本地訓練／選模範圍不能只靠資料集名稱視為一致 | 披露 CLIP 凍結、checkpoint parent、既有選模重疊和開發用途 |

**PAPER FACT** 的來源是 [3experiments.tex](paper/metafind_source/3experiments.tex)：第 15 行 mean pooling、第 18 行指標、第 24 行七模態、PC-only baseline 與 Stage 2 在 Objaverse 關閉 layout 的說明。
新主評估與逐條件人工 qrels 是已批准的 **IMPLEMENTATION CHOICE**；作者完整 query／gallery／正解協定仍為 **UNKNOWN**。
需求描述、模型或規則若在查看排名後修改，須記錄新版本並視為開發比較，不能繼續宣稱該題組未參與選擇。

## 輸入格式

準備一份 JSON spec。以下僅為**人工格式範例**，其中需求、`asset_a/b/c`、審核者及 qrels 都是占位示意，必須換成真實來源與已完成的審核；不可直接當成正式題庫。

```json
{
  "schema": "metafind.intent_retrieval.spec.v1",
  "gallery_uids": ["asset_a", "asset_b", "asset_c"],
  "provenance": {
    "source": "FIXTURE ONLY: replace with the declared candidate-pool source",
    "creator": "FIXTURE ONLY: replace with the actual curator",
    "method": "FIXTURE ONLY: replace with the documented selection method"
  },
  "queries": [
    {
      "query_id": "need-example-001",
      "text": "A compact dining chair with a wooden appearance.",
      "conditions": ["text"],
      "provenance": {
        "source": "FIXTURE ONLY: replace with the real request source",
        "creator": "FIXTURE ONLY: replace with the actual author",
        "method": "FIXTURE ONLY: independently written request"
      },
      "qrels": {"text": {"asset_a": 1, "asset_b": 1, "asset_c": 0}},
      "judgments": {
        "text": {
          "assessor": "FIXTURE ONLY: replace with the actual assessor",
          "method": "FIXTURE ONLY: replace with the actual review procedure",
          "criteria": "FIXTURE ONLY: replace with the explicit acceptance criteria"
        }
      }
    }
  ]
}
```

| 欄位 | 契約 |
|---|---|
| `gallery_uids` | 固定、無重複的真實候選 UID；每題 qrels 都覆蓋全部候選 |
| `queries[].query_id` | 非空且不重複的需求編號；不要求出現在 gallery |
| `text` | 提供 T 條件時的實際需求文字 |
| `images` | 可選的原始圖片路徑列表；有 I 的條件必須提供 |
| `pointcloud` | 可選 `.npy` 路徑；有 P 的條件必須提供 raw float32、shape `(10000, 6)` 的 xyzrgb |
| `conditions` | 從上述七個精確名稱選擇；所需原始輸入必須存在 |
| query `provenance` | `source`、`creator`、`method`，記錄需求與參考資料如何產生 |
| `qrels` | 每個宣告條件各有一份全候選 `UID → 0/1`；不能借用別條件或把漏標填零 |
| `judgments` | 每個宣告條件的 `assessor`、`method`、`criteria` |

相對圖片／點雲路徑以 spec 所在目錄為基準。prepare 核對真實圖片可解碼，保存原始圖片／點雲來源與 hash，不接受預先算好的外部 embedding。
參考照片由本次模型的 preprocess 與 image encoder 處理；一題提供多張時，採各張特徵的算術平均，不把圖片拼成一張。候選圖片則使用已綁定來源的全部快取視角之 float32 算術平均。
點雲原檔保持不變；runner 對 query xyz 採既有 `pc_norm`，以 float64 計算後轉 float32，RGB 保持原值，再使用本次正確模型編碼。全部點落在同一位置的退化點雲不能使用。

## 執行模板

下列命令是操作模板，請將 `/path/to/...` 換成真實路徑。**先完成並封存人工題目與 qrels，才執行模型評估。**
`METAFIND_DATA` 要指向與候選資料及 checkpoint encoding 相容的資料根目錄；不要因檔名相同就假設內容一致。

```bash
cd /home/kyzen/MetaFindV1
export METAFIND_DATA=/path/to/evaluation_data
PY=/home/kyzen/miniconda3/envs/MetaFind/bin/python

# 1. 凍結真實需求、逐條件 qrels、候選與原始參考來源。
$PY -m metafind.eval.intent_protocol \
  --spec /path/to/intent_spec.json \
  --out-dir /path/to/intent_protocol_run01

# 2. 核對輸入與 checkpoint；此步不產生檢索分數。
$PY -m metafind.eval.intent_retrieval \
  --protocol /path/to/intent_protocol_run01/protocol.json \
  --stage1-record /path/to/stage1_best_ckpt.json \
  --stage2-record /path/to/variant_ckpts.json \
  --stage2-variant full \
  --out-dir /path/to/intent_eval_run01 \
  --check-only

# 3. 通過必要核對且有可用資源後，另行執行模型。
$PY -m metafind.eval.intent_retrieval \
  --protocol /path/to/intent_protocol_run01/protocol.json \
  --stage1-record /path/to/stage1_best_ckpt.json \
  --stage2-record /path/to/variant_ckpts.json \
  --stage2-variant full \
  --out-dir /path/to/intent_eval_run01 \
  --device cpu --batch-size 16 --block 4096 --seed 20260908
```

只比較 mean／Stage 1 時省略 `--stage2-record`。CPU 命令是明確的執行模板，不是完整大池效能承諾；改用 GPU 前仍須安排可用資源，不干擾執行中的標註或訓練。
輸出目錄使用新的名稱；失敗記錄保留，不把中途部分結果當作完整表格。
`--check-only` 不代表人工判斷正確、模型一定有檢索能力，或完整模型 forward 已執行。

## 評估產物與能證明的範圍

- `protocol.json`：凍結的需求、逐條件 qrels、候選清單及原始來源身分。
- `provenance.json`：模型 parent／checkpoint、程式版本、執行指令、裝置、seed 與預處理規則。
- `results.json`、`table.md`：各模型／條件的 Hit 與 Recall、題數；候選總數另記在 `results.json` 的 `n_gallery`。
- `<method>__<condition>.jsonl`：逐題排名、命中與正例資訊；組合條件在檔名以底線取代加號。
- `ulip2_query_tokens.json`、`stage1_query_tokens.json`：真正編碼時的文字與有效 token IDs；`--check-only` 不產生這項證據。
- `failure.json`：模型執行失敗的原因，不當成完整成績。

目前結果標記為 `development_comparison`、`independent_test_status: not_certified`，不自動認證獨立測試。
來源 hash 能核對輸入是否漂移，不能證明審核者從未看過模型輸出，也不能證明人工判準正確；這些需要真實的審核流程紀錄。
候選影像的快取 bytes 與側車會綁定；若歷史快取缺少原始 CLIP 權重身分，不能用現在的模型版本追溯證明當時的 encoder。

## 標註結束後的執行順序

1. 整理有效標註、真正失敗與已批准的人工排除，保留各自清單與來源。
2. 建立文字／圖片特徵與資料切分；G3 讀取切分及協定後另行執行，不把預檢放在其輸入產生之前。
3. 訓練 Stage 1，記錄 checkpoint 選擇使用的資料；建立並核驗其候選向量。
4. 準備 ProcTHOR 物件、房間圖、節點／語義關係與候選資料，訓練 Stage 2；固定父子 checkpoint 身分。
5. 撰寫真實需求、取得原始參考線索，先固定候選池，再完成每個可見模態條件的人工 qrels，於模型排名前封存。此工作可以提早並行準備。
6. 另行執行主評估，報告 Hit@1／Hit@5、輔助 Recall、逐條件題數與候選數；舊同資產辨識結果另外標示為診斷。

標註結束不會自動產生需求或完成人工審核。這條主評估也不自動完成場景品質、裁判協定或論文 Table 2；相關邊界見 [場景分數輸入與彙總](SCENE_SCORES.md)。

2026-09-08 實作、CPU 驗證與真實 ULIP-2 椅子草稿的證據見[本次更新紀錄](audit/INTENT_RETRIEVAL_UPDATE_20260908.md)。草稿已抽出 BLIP 原文與點雲，尚未完成需求正解審核，不能當成正式評估成績。

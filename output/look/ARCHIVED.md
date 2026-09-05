> **2026-09-05 17:16 更新**：下面提到的 `/mnt/data1/kyzen/MetaFindV1_archive/…` 與 `archive_20260902_pre11view/…` 已依 Kyzen 指示（「archive 刪掉」）**整個刪除**。這裡列的是當時搬走的內容清單，檔案已不存在。

# 已歸檔（2026-09-05，Kyzen「output/look 沒整理啊」）

搬到 `/mnt/data1/kyzen/MetaFindV1_archive/output_look_20260905/`，原檔不刪：

- `slides/slide-01..21.png`：09-04 簡報的逐頁預覽圖（可由 `MetaFind_report_20260904.pptx` 重新輸出）。
- `reports/MetaFind_report_20260904_plain.{pptx,pdf}`：同一份簡報的無圖版。
- `probes/*.json`（20 個）：ledger、docs、程式碼、測試**都沒有引用**的舊探針輸出：
  exp_derange_{image,pc,text}、exp_image_{disjoint_views,eleven_views,four_views,single_view}、
  exp_obs_x_gallery、exp_prefusion_norm、exp_similarity_dot、exp_text_figure2、exp_text_x_gallery、
  exp_text_x_singleview、exp_type_level_query_thumbnail、exp_ulip_image_{disjoint_views,single_view}、
  exp_ulip_table1_full、exp_ulip_text_{cat_only,desc_only,fill0}。
- 刪掉的唯一檔案：`pptx/slides/output/MetaFind_report_20260904.pptx`，與上一層同名檔 md5 相同（2921fb3a…）。

更早（2026-09-02 之前）的 88 份在 `/mnt/data1/kyzen/archive_20260902_pre11view/look/`。
- `pptx_build_previews/`：`pptx/slides/output/` 裡 41 張建置預覽圖（s-*.png、chk*.png、contact.png），可重建。

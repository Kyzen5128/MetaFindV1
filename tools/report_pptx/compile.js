// MetaFind 重現進度報告 2026-09-04 -- built with the pptx-generator skill (PptxGenJS).
// Numbers are the same as docs/NOTE_20260904_* and the Notion report note; charts from
// tools/probes/draw_report_charts.py (copied into ./imgs).
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";              // 10" x 5.625"

// Palette 10 "Education & Charts": deep slate, teal, sand, orange, coral.
const theme = { primary: "264653", secondary: "2a9d8f", accent: "e76f51", light: "e9c46a", bg: "fbfaf6" };
const ZH = "Microsoft YaHei", EN = "Arial";
const INK = "1f2a30", GREY = "5c6b73", CARD = "ffffff", LINE = "d9d4c7";

// ---------------------------------------------------------------- helpers
let pageNo = 1;
function badge(slide) {
  pageNo += 1;
  slide.addShape(pres.shapes.OVAL, { x: 9.3, y: 5.1, w: 0.4, h: 0.4, fill: { color: theme.secondary } });
  slide.addText(String(pageNo), { x: 9.3, y: 5.1, w: 0.4, h: 0.4, fontSize: 11, fontFace: EN, color: "FFFFFF", bold: true, align: "center", valign: "middle" });
}
function base(title, subtitle) {
  const s = pres.addSlide();
  s.background = { color: theme.bg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.18, h: 5.625, fill: { color: theme.primary } });
  s.addText(title, { x: 0.45, y: 0.22, w: 8.8, h: 0.6, fontSize: 24, fontFace: ZH, color: theme.primary, bold: true, margin: 0, fit: "shrink" });
  if (subtitle) s.addText(subtitle, { x: 0.45, y: 0.78, w: 9.1, h: 0.32, fontSize: 11, fontFace: ZH, color: GREY, margin: 0, fit: "shrink" });
  badge(s);
  return s;
}
function bullets(s, items, box, size) {
  size = size || 12;
  const runs = items.map((it) => {
    const lvl = Array.isArray(it) ? it[0] : 0, txt = Array.isArray(it) ? it[1] : it;
    const bold = txt.startsWith("**"); const t = txt.replace(/\*\*/g, "");
    return { text: t, options: { bullet: lvl === 0 ? { indent: 12 } : { indent: 12, code: "2013" }, indentLevel: lvl, fontSize: lvl ? size - 1 : size, fontFace: ZH, color: lvl ? GREY : INK, bold: bold, paraSpaceAfter: 5, breakLine: true } };
  });
  s.addText(runs, Object.assign({ fontFace: ZH, valign: "top", margin: 2, fit: "shrink" }, box));
}
function table(s, rows, box, opts) {
  opts = opts || {};
  const fs = opts.fontSize || 9;
  const data = rows.map((r, i) => r.map((c) => {
    const txt = String(c); const bold = txt.startsWith("**"); const t = txt.replace(/\*\*/g, "");
    return { text: t, options: { fontSize: fs, fontFace: ZH, color: i === 0 ? "FFFFFF" : INK, bold: i === 0 || bold, fill: { color: i === 0 ? theme.primary : (i % 2 ? "FFFFFF" : "f3efe6") }, valign: "middle", margin: 0.03 } };
  }));
  s.addTable(data, Object.assign({ border: { type: "solid", pt: 0.5, color: LINE }, colW: opts.colW, rowH: opts.rowH || 0.26, autoPage: false }, box));
}
function card(s, box, title, body, color) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, Object.assign({ fill: { color: CARD }, line: { color: LINE, pt: 0.75 }, rectRadius: 0.08 }, box));
  s.addShape(pres.shapes.RECTANGLE, { x: box.x, y: box.y, w: 0.08, h: box.h, fill: { color: color || theme.secondary } });
  s.addText(title, { x: box.x + 0.18, y: box.y + 0.08, w: box.w - 0.3, h: 0.32, fontSize: 12, fontFace: ZH, color: theme.primary, bold: true, margin: 0 });
  s.addText(body, { x: box.x + 0.18, y: box.y + 0.42, w: box.w - 0.3, h: box.h - 0.5, fontSize: 10, fontFace: ZH, color: INK, valign: "top", margin: 0, fit: "shrink" });
}
function note(s, text, y) {
  s.addText(text, { x: 0.45, y: y || 5.05, w: 8.6, h: 0.45, fontSize: 9.5, fontFace: ZH, color: GREY, margin: 0, valign: "top", fit: "shrink" });
}
function section(num, title, intro) {
  const s = pres.addSlide();
  s.background = { color: theme.primary };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 3.2, h: 5.625, fill: { color: theme.secondary } });
  s.addText(num, { x: 0.4, y: 1.6, w: 2.5, h: 1.6, fontSize: 88, fontFace: EN, color: "FFFFFF", bold: true, margin: 0 });
  s.addText(title, { x: 3.7, y: 1.9, w: 5.8, h: 0.9, fontSize: 30, fontFace: ZH, color: "FFFFFF", bold: true, margin: 0, fit: "shrink" });
  if (intro) s.addText(intro, { x: 3.7, y: 2.85, w: 5.8, h: 1.2, fontSize: 14, fontFace: ZH, color: "e9e4d8", margin: 0, valign: "top", fit: "shrink" });
  badge(s);
}
const IMG_DIMS = {"report_fig2_level_shape.png": [1170, 624], "report_fig4_query_pc_observation.png": [1560, 780], "report_fig1_stage1_arms_D.png": [1560, 780], "report_fig6_ulip2_zero_shot.png": [845, 546], "ulip2_pull_explainer.png": [1820, 1430], "report_fig3_P1_epochs.png": [1300, 715], "report_fig5_stage2.png": [1820, 715]};
function img(s, file, box) {
  const [pw, ph] = IMG_DIMS[file]; const ar = pw / ph;
  let w = box.w, h = box.h;
  if (!w) w = h * ar;
  if (!h) h = w / ar;
  const maxW = box.maxW || (9.7 - box.x);
  if (w > maxW) { w = maxW; h = w / ar; }
  s.addImage({ path: "./imgs/" + file, x: box.x, y: box.y, w: w, h: h });
}

// ================================================================ 1 cover
{
  const s = pres.addSlide();
  s.background = { color: theme.primary };
  s.addShape(pres.shapes.RECTANGLE, { x: 6.3, y: 0, w: 3.7, h: 5.625, fill: { color: theme.secondary } });
  s.addShape(pres.shapes.OVAL, { x: 6.9, y: 1.3, w: 2.6, h: 2.6, fill: { color: theme.light }, line: { color: theme.light } });
  s.addText("MetaFind", { x: 6.9, y: 2.1, w: 2.6, h: 1.0, fontSize: 30, fontFace: EN, color: theme.primary, bold: true, align: "center", valign: "middle", margin: 0 });
  s.addText("MetaFind 重現進度報告", { x: 0.6, y: 1.5, w: 5.6, h: 1.0, fontSize: 36, fontFace: ZH, color: "FFFFFF", bold: true, margin: 0, fit: "shrink" });
  s.addText("Stage 1 做了什麼測試、目前數據與現況、之前錯在哪與原因、做了什麼改變、Stage 2 做了什麼、接下來計畫", { x: 0.6, y: 2.6, w: 5.4, h: 1.1, fontSize: 14, fontFace: ZH, color: "e9e4d8", margin: 0, valign: "top" });
  s.addText("2026-09-04", { x: 0.6, y: 4.5, w: 3, h: 0.4, fontSize: 12, fontFace: EN, color: theme.light, margin: 0 });
}

// ================================================================ 2 TOC
{
  const s = pres.addSlide(); s.background = { color: theme.bg };
  s.addText("大綱", { x: 0.6, y: 0.4, w: 5, h: 0.7, fontSize: 30, fontFace: ZH, color: theme.primary, bold: true, margin: 0 });
  const items = [["01", "MetaFind 與重現目標", "架構、公式、資料、評估協定"],
                 ["02", "之前錯在哪、原因、改了什麼", "同一份紀錄的平凡解、文字模板、正規化、工具 bug"],
                 ["03", "Stage 1 測試與目前數據", "八個版本、14 格、為什麼選 P1、ULIP-2 檢驗、融合格為什麼高"],
                 ["04", "Stage 2 做了什麼", "設計與配方、四次跑、S2-C 候選"],
                 ["05", "現況與接下來計畫", "五段排程、判斷規則、待決定"]];
  items.forEach((it, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 4.7, y = 1.4 + row * 1.25;
    s.addText(it[0], { x: x, y: y, w: 0.9, h: 0.8, fontSize: 30, fontFace: EN, color: theme.accent, bold: true, margin: 0 });
    s.addText(it[1], { x: x + 0.95, y: y, w: 3.6, h: 0.42, fontSize: 15, fontFace: ZH, color: theme.primary, bold: true, margin: 0 });
    s.addText(it[2], { x: x + 0.95, y: y + 0.42, w: 3.6, h: 0.5, fontSize: 10, fontFace: ZH, color: GREY, margin: 0, valign: "top" });
  });
  badge(s);
}

// ================================================================ section 01
section("01", "MetaFind 與重現目標", "雙塔 + ULIP-2 + Fusion；Stage 1 物件層級對比學習，Stage 2 加房間佈局。目標：論文 Table 1。");

// 3 architecture
{
  const s = base("MetaFind 架構回顧", "論文 §2.2、§2.4、§2.6（PAPER FACT）");
  // simple flow diagram on the left
  const boxes = [["query 塔", "text / image / pc 任意子集；缺的放 mask token", 0.45, 1.3, theme.secondary],
                 ["ULIP-2 backbone", "CLIP 文字塔、CLIP 影像塔、Point-BERT，各出 1280 維", 0.45, 2.35, theme.primary],
                 ["Fusion Transformer", "每塔一個；Stage 2 再加 λ·e_layout（ESSGNN）", 0.45, 3.4, theme.primary],
                 ["gallery 塔", "三模態齊全；預訓完凍結，向量快取", 0.45, 4.45, theme.accent]];
  boxes.forEach((b) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: b[2], y: b[3], w: 4.0, h: 0.8, fill: { color: b[4] }, line: { color: b[4] }, rectRadius: 0.08 });
    s.addText([{ text: b[0], options: { bold: true, fontSize: 12, breakLine: true } }, { text: b[1], options: { fontSize: 9 } }], { x: b[2] + 0.15, y: b[3], w: 3.7, h: 0.8, fontFace: ZH, color: "FFFFFF", valign: "middle", margin: 0 });
  });
  s.addText("sim(e_query, e_gallery) = cosine ／ τ = 0.5 → R@1 / R@5", { x: 0.45, y: 5.28, w: 4.2, h: 0.3, fontSize: 9, fontFace: ZH, color: GREY, margin: 0 });
  bullets(s, ["**Stage 1（Objaverse-LVIS）：兩塔都訓；query 每個模態獨立 30% 遮掉；Eq. 5 單向 InfoNCE",
              [1, "L = −log exp(sim(f_q(Q), f_g(A))/τ) / Σ_{A'∈B} exp(sim(f_q(Q), f_g(A'))/τ)，τ = 0.5"],
              "**Stage 2（ProcTHOR）：凍結 gallery 與 ULIP 編碼器；e_query = Fusion + λ·e_layout（Eq. 6）；雙向 Eq. 7/8；30% batch 不給 layout",
              "**Table 1 的 w/ ESSGNN 列：Stage 2 的 query 頭在 Objaverse 上評、layout 關掉（§3.2）",
              "**論文沒寫：lr、epochs、batch、優化器、warmup、選 checkpoint 規則、Transformer 深度、query 的三份觀測怎麼來"],
          { x: 4.75, y: 1.25, w: 4.9, h: 4.2 }, 11);
}

// 4 data & eval
{
  const s = base("我們的資料與評估", "只有資料集允許先不一致；其餘訓練設計要與論文相同");
  table(s, [["項目", "我們", "論文", "分類"],
            ["資產數", "45,692（官方 LVIS 清單 46,052，扣掉點雲／渲染／標註缺的）", "約 48,000", "DEVIATION"],
            ["視角", "12（OpenShape 三圈 × 4）", "11 個正交", "DEVIATION；11-of-12 量過指紋不動"],
            ["點雲", "10,000 點 xyzrgb，ULIP 正規化", "同 ULIP-2", "UPSTREAM FACT"],
            ["標註", "gemma-4-12B-it，13 個欄位（Figure 2）", "GPT-4o", "DEVIATION"],
            ["切分", "72/8/20：dev_train 31,985、dev_val 4,569、test 9,138（封存）", "80/20", "多切一塊驗證用"],
            ["評估器", "float64 cosine，同分算模型輸", "—", "OBSERVED"],
            ["協定 C", "4,569 dev_val query 對 4,569 dev_val gallery（選 checkpoint）", "—", ""],
            ["協定 D", "4,569 dev_val query 對 36,554 train gallery（主要比對表）", "gallery 大小沒寫", ""]],
        { x: 0.45, y: 1.2, w: 9.1 }, { colW: [0.9, 4.6, 1.3, 2.3], fontSize: 9 });
  card(s, { x: 0.45, y: 3.75, w: 9.1, h: 1.2 }, "目標列：論文 w/o ESSGNN（R@1 / R@5）",
       "text 13.8 / 23.1　image 11.7 / 19.2　pc 75.1 / 78.0　T+I 17.2 / 21.8　T+PC 44.5 / 71.3　I+PC 45.8 / 73.1　full 51.7 / 76.5\n形狀特徵：pc 一個人最高；加了 text / image 反而掉。", theme.accent);
}

// ================================================================ section 02
section("02", "之前錯在哪、原因、改了什麼", "七個問題，其中一個決定一切：query 跟 gallery 是同一份紀錄。");

// 5 main error
{
  const s = base("之前錯在哪（一）：query 讀的是 gallery 同一份紀錄", "E1，EXPERIMENTALLY DISFAVORED");
  bullets(s, ["**現象：舊構法 pilot10b 七格全部 90 以上，full 100.0；論文 full 只有 51.7",
              "**原因：query 塔和 gallery 塔吃同一個 dict、同一批張量（點雲也是同一次 Point-BERT forward）",
              "**Eq. 5 在 query ≡ gallery 時有平凡解 f_query ≡ f_gallery。實測 cos(q, g) = 0.9989",
              "**論文 full 51.7，所以論文不可能這樣餵。這不是定理，是實驗上的否定",
              "**修正：query 改讀「第二份觀測」：影像用單一視角（P1），文字用另一段描述、點雲重取樣（P5）",
              "**還沒完：重取樣點雲跟原檔 cos 0.99、文字同一句。所以 P1 的融合格仍偏高（§03 說明）"],
          { x: 0.45, y: 1.25, w: 5.6, h: 4.1 }, 12);
  card(s, { x: 6.3, y: 1.3, w: 3.3, h: 1.7 }, "舊構法 pilot10b（D 協定 R@1）", "text 58.0　image 84.6　pc 78.8\nT+I 96.5　T+PC 99.6　I+PC 94.1\nfull 100.0", theme.accent);
  card(s, { x: 6.3, y: 3.2, w: 3.3, h: 1.7 }, "論文 w/o ESSGNN", "text 13.8　image 11.7　pc 75.1\nT+I 17.2　T+PC 44.5　I+PC 45.8\nfull 51.7", theme.secondary);
}

// 6 other errors
{
  const s = base("之前錯在哪（二）：其他六項");
  table(s, [["#", "之前", "為什麼錯", "證據", "修正"],
            ["2", "文字字串把 160 字自由描述放最前面", "Figure 2 / Figure 1 的文字輸入是類別＋結構化欄位的短填表", "text 58.0 → 8.3", "填表模板 attrs_v1：{category} made of {materials}, roughly {w}×{l}×{h} cm, {placement}"],
            ["3", "進 Fusion 前不正規化", "訓練後 pc 範數 139、text 37、image 40，點雲主導注意力", "C8", "每個模態先 L2 再進 Transformer"],
            ["4", "probe 工具把 gallery 的 pc 餵成零", "零張量不是缺席，結論全錯", "抓到後全撤回", "工具改拒跑，parity test 釘住"],
            ["5", "Stage 2 builder 漏帶 prefusion_norm / image_tokens", "Stage 2 的塔跟父不是同一設定", "F1", "修好並加測試"],
            ["6", "Stage 2 批次尾巴退化（B=1 → loss 0）", "唯一正例批次的尾巴", "F2", "少於 8 個樣本的批次丟掉"],
            ["7", "τ、scorer、tie 規則各處不一致", "數字比不了", "", "全部走同一個評估器 metafind/eval/retrieval.py"]],
        { x: 0.45, y: 1.2, w: 9.1 }, { colW: [0.3, 2.3, 2.6, 1.1, 2.8], fontSize: 8.5, rowH: 0.5 });
}

// 7 what changed
{
  const s = base("做了什麼改變（總表）");
  table(s, [["改變", "對應論文哪句", "機制", "分類"],
            ["query 讀第二份觀測", "Table 1 full 51.7（同一份會 100）", "--query-image-policy single_view；query pack（另一段描述、重取樣點雲）", "PAPER-CONSTRAINED"],
            ["填表文字", "Figure 2 的 13 個欄位、Figure 1 的文字輸入", "TEXT_TEMPLATES attrs_v1；文字快取重編", "PAPER FIGURE FACT"],
            ["融合前 L2", "論文沒寫；量到 pc 主導", "FusionConfig.prefusion_norm", "IMPLEMENTATION CHOICE"],
            ["單一評估器", "R@1 / R@5", "float64 cosine，同分算輸；level / shape 兩個距離", "OBSERVED"],
            ["Stage 2 塔對齊父", "§2.6 凍結 gallery、只訓 query Fusion + ESSGNN", "build_stage2_model 鏡射 Stage 1 FusionConfig", "bug fix"],
            ["Stage 2 配方", "§2.6 叫 fine-tuning，沒給 lr", "lr 5e-5、warmup 10%、cosine；query 遮罩模式可選", "IMPLEMENTATION CHOICE"],
            ["query 點雲可換觀測", "論文沒寫 query 點雲怎麼來", "--query-pc-perturb（單邊掃描／去色／雜訊…），記錄在 run", "IMPLEMENTATION CHOICE（實驗中）"]],
        { x: 0.45, y: 1.2, w: 9.1 }, { colW: [1.6, 2.7, 3.4, 1.4], fontSize: 8.5, rowH: 0.42 });
}

// ================================================================ section 03
section("03", "Stage 1 測試與目前數據", "八個版本、對 Table 1 的 14 格、為什麼選 P1、ULIP-2 有沒有被動、融合格為什麼高、訓練夠不夠。");

// 8 designs
{
  const s = base("Stage 1 做了哪些測試：八個版本、共同設定");
  table(s, [["版本", "跟前一版差在哪", "終點 loss", "dev_val 平均 R@1"],
            ["pilot10b", "舊構法：query = gallery 同一份、長描述模板、12 視角平均、不正規化", "2.34", "0.947"],
            ["**P1", "填表文字 + query 單一視角 + 融合前 L2", "2.47", "0.795"],
            ["P3", "P1 + 12 個視角各一個 token 進 Fusion", "2.49", "0.767"],
            ["P4", "P1 + 兩塔共用一份 Fusion", "2.41", "0.746"],
            ["P5", "描述文字；query = 另一段描述 + 重取樣點雲 + 單視角", "2.44", "0.839"],
            ["P6", "P1 但每步隨機抽視角", "2.48", "0.795"],
            ["P7", "P1 但關掉融合前 L2", "2.41", "0.788"]],
        { x: 0.45, y: 1.15, w: 9.1 }, { colW: [0.9, 5.6, 1.1, 1.5], fontSize: 9 });
  bullets(s, ["共同設定：dev_train 31,985 / dev_val 4,569；AdamW lr 5e-4、wd 0.1、betas (0.9, 0.98)、warmup 1 epoch、cosine；batch 64；10 epochs；τ 0.5；30% 遮罩；seed 20260816",
              "可訓練：兩座 Fusion + Point-BERT + pc_projection；CLIP 文字／影像塔凍結（論文 §3.4 說 full encoder fine-tuning 較好 → 唯一的設計偏離，記憶體限制）",
              "選 checkpoint：每代在 dev_val 測七格 R@1 平均取最高（都是第 9 代）；ULIP 官方也是每代驗證取最好"],
          { x: 0.45, y: 3.45, w: 9.1, h: 1.6 }, 10);
}

// 9 results table
{
  const s = base("Stage 1 結果：對 Table 1 的 14 格（D 協定，R@1）");
  table(s, [["版本", "text", "image", "pc", "T+I", "T+PC", "I+PC", "full", "level", "shape"],
            ["**論文", "**13.8", "**11.7", "**75.1", "**17.2", "**44.5", "**45.8", "**51.7", "0", "0"],
            ["P1", "11.6", "29.7", "66.6", "67.5", "95.6", "77.8", "98.1", "0.59", "0.41"],
            ["P3", "10.4", "24.6", "61.1", "59.7", "94.6", "72.7", "98.0", "0.56", "0.40"],
            ["P4", "12.0", "25.0", "52.3", "58.5", "94.4", "65.8", "98.0", "0.54", "0.41"],
            ["P5", "14.3", "49.4", "88.5", "59.5", "92.8", "94.0", "95.5", "0.66", "0.40"],
            ["P6", "12.8", "28.8", "67.3", "67.1", "96.9", "77.3", "98.2", "0.58", "0.40"],
            ["P7", "9.5", "36.1", "69.1", "64.6", "95.6", "81.7", "98.5", "0.61", "0.44"],
            ["pilot10b", "58.0", "84.6", "78.8", "96.5", "99.6", "94.1", "100.0", "0.91", "0.57"]],
        { x: 0.45, y: 1.15, w: 9.1 }, { colW: [1.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.85, 0.85], fontSize: 10, rowH: 0.3 });
  card(s, { x: 0.45, y: 4.0, w: 4.4, h: 1.15 }, "兩個距離", "level = 14 格 |ln(我們/論文)| 的平均（整體高低）\nshape = 扣掉整體高低後只看七格的相對形狀。越小越像。", theme.secondary);
  card(s, { x: 5.15, y: 4.0, w: 4.4, h: 1.15 }, "讀法", "單模態格已貼近論文（text 11.6 對 13.8、pc 66.6 對 75.1）。跑偏的全是含 pc 的融合格：論文 pc 75 → T+PC 44（掉），我們 67 → 96（升）。", theme.accent);
}

// 10 fig1
{ const s = base("圖 1：七個版本 vs 論文（D 協定）"); img(s, "report_fig1_stage1_arms_D.png", { x: 0.9, y: 1.15, h: 4.2 }); }

// 11 fig2
{
  const s = base("圖 2：離論文多遠；架構軸沒有一個能改變形狀");
  img(s, "report_fig2_level_shape.png", { x: 0.45, y: 1.2, h: 3.6 });
  bullets(s, ["七個新版本 shape 都在 0.40～0.44，舊構法 0.57", "Fusion 一份／兩份、token／平均、正規化開／關、視角固定／隨機：形狀都不動", "結論：差距不在架構，在 query 的觀測跟 gallery 太像"], { x: 7.3, y: 1.3, w: 2.4, h: 3.5 }, 11);
}

// 12 why P1
{
  const s = base("為什麼選 P1");
  const cards = [["1. 乾淨主線", "一次只動一個變數；三個修正各對應論文一句話：Figure 2 填表、Figure 1 單張圖、C8 正規化"],
                 ["2. 單模態最接近", "text 11.6 對 13.8；pc 66.6 對 75.1"],
                 ["3. 其他沒更好", "P4 的 pc 掉到 52；P3 / P6 / P7 同族；P5 各格更高（重取樣點雲 cos 0.99 還是太像）"],
                 ["4. 可被推翻", "主線 = Stage 2 的父。已授權：lr、epochs、query 觀測都在重跑"]];
  cards.forEach((c, i) => card(s, { x: 0.45 + (i % 2) * 4.65, y: 1.25 + Math.floor(i / 2) * 1.95, w: 4.45, h: 1.75 }, c[0], c[1], i % 2 ? theme.accent : theme.secondary));
}

// 13 ULIP-2 integrity
{
  const s = base("ULIP-2 沒有被動", "懷疑：融合格「越加越高」像三個準度相加，是不是 ULIP-2 被改了？");
  img(s, "report_fig6_ulip2_zero_shot.png", { x: 0.45, y: 1.2, h: 3.4 });
  bullets(s, ["vendored ULIP 17 個檔案與上游逐 byte 相同；相容補丁只補 torch._six、純 torch 最遠點取樣（同演算法）、knn 替身",
              "釋出權重零樣本分類 Objaverse-LVIS，照官方 main.py 流程（64 模板 → 正規化 → 平均 → 再正規化，cosine，1,156 類）",
              "**top-1 50.9 / top-5 79.3；論文 Table 10 是 50.6 / 79.1",
              "結論：backbone、點雲前處理、文字塔就是論文的 ULIP-2。分數高不是這裡來的"], { x: 5.95, y: 1.25, w: 3.7, h: 3.8 }, 11);
}

// 14 explainer figure
{ const s = base("融合格為什麼高：query 跟 gallery 有兩個模態是同一份檔"); img(s, "ulip2_pull_explainer.png", { x: 1.5, y: 1.1, h: 4.45 }); }

// 15 in numbers
{
  const s = base("融合格為什麼高（用數字講）", "釋出 ULIP-2，1,024 個 dev_val 物件實測");
  bullets(s, ["**ULIP-2 的「拉近」= 自己的模態比別人的近，不是三點疊成一點：cos(text, 自己 pc) 0.29、cos(image, 自己 pc) 0.46；對別人只有 0.07 / 0.11",
              "**query 的 pc 跟 gallery 的 pc 是同一個檔（cos 1.00），最近的別人 0.59",
              "**加 text：query 離自己 0.80、離別人 0.53。加 text + image：0.76 對 0.52。差距一直在，排名翻不了",
              "**不用 Fusion、不訓練、單純平均也一樣往上（T+PC 98.6）。不是 Fusion 學出「篩選線索」，是資料讓它只能往上",
              "**P1 的 pc-only 66.6 不是 100：gallery 是三模態融合向量，pc-only 的 query 是 pc + 兩個 mask token，走的路不一樣；full 三模態齊全走同一條路，回到 98",
              "**論文 full 51.7 < pc 75.1：代表論文 query 的三份觀測跟 gallery 不是同一份，而且 text / image 會偏向別人。論文沒寫怎麼取得"],
          { x: 0.45, y: 1.2, w: 9.1, h: 4.1 }, 11.5);
}

// 16 query pc observation
{
  const s = base("只換 query 的點雲觀測（P1 直接重評，不重訓）");
  img(s, "report_fig4_query_pc_observation.png", { x: 0.3, y: 1.15, h: 3.3 });
  table(s, [["query pc", "cos", "pc", "T+PC", "I+PC", "full"],
            ["原檔", "1.00", "66.6", "95.6", "77.8", "98.1"], ["重取樣", "0.997", "63.8", "94.6", "76.2", "97.3"],
            ["去顏色", "0.83", "8.9", "20.4", "14.1", "31.4"], ["加雜訊", "0.82", "9.1", "38.8", "23.8", "58.7"],
            ["單邊掃描", "0.81", "5.4", "28.0", "17.2", "46.6"], ["**論文", "?", "**75.1", "**44.5", "**45.8", "**51.7"]],
        { x: 6.95, y: 1.2, w: 2.75 }, { colW: [0.75, 0.4, 0.4, 0.4, 0.4, 0.4], fontSize: 8, rowH: 0.24 });
  note(s, "P1 的點雲塔很脆（cos 0.82 就掉到個位數；釋出的 ULIP-2 同樣雜訊還有 45），而且不管怎麼弄 full 都高於 pc。光弄壞 pc 做不出論文形狀，text / image 也得是不同觀測 → P8 實驗。", 4.6);
}

// 17 training enough
{
  const s = base("訓練夠不夠？", "論文沒寫 epochs；250 是 ULIP 程式碼的預設值（main.py:23），不是 MetaFind 說的");
  img(s, "report_fig3_P1_epochs.png", { x: 0.3, y: 1.15, h: 3.6 });
  bullets(s, ["P1 從第 0 代起 full 就高於 pc；訓練越久七格一起往上", "舊版本跑 44 代：七格全 90 以上",
              "ULIP 官方：固定 250 代 cosine，每代在 ModelNet40 驗證取 best，沒有 early stopping。我們選法相同，只差代數",
              "**驗證中：P1 25 代（跑中）；lr 掃描 1e-4 / 1e-3 / 3e-3（排隊）"], { x: 6.95, y: 1.25, w: 2.75, h: 3.8 }, 10);
}

// ================================================================ section 04
section("04", "Stage 2 做了什麼", "ESSGNN 佈局向量、ProcTHOR 房間層級微調、Table 1 的 w/ ESSGNN 列。");

// 18 stage 2 design
{
  const s = base("Stage 2 設計與配方", "審計：Eq. 2/3/4、6/7/8、附錄 C 逐項對過；正文與附錄矛盾三處走附錄版");
  table(s, [["項目", "值", "分類"],
            ["場景", "ProcTHOR-10k train 前 1,500 間（全量 9,600 尚未跑）；切分 9,600/2,400（論文 80/20）vs 官方 10k/1k/1k 待定", "IMPLEMENTATION CHOICE"],
            ["樣本", "99,945 個（房間內每個物件當一次 query）", "OBSERVED"],
            ["gallery", "1,439 個 ProcTHOR 資產，由 P1 編碼", "OBSERVED"],
            ["凍結", "gallery 塔、ULIP 編碼器；只訓 query Fusion + ESSGNN", "PAPER FACT §2.6"],
            ["λ 初值", "0.1 × median‖Fusion‖ = 93.46", "IMPLEMENTATION CHOICE"],
            ["場景 dropout", "每 batch 30% 不給 layout", "PAPER FACT"],
            ["損失", "Eq. 7/8 雙向，取平均，τ 0.5", "PAPER FACT"],
            ["配方", "先導：Stage 1 配方平坦 5e-4；S2-C / S2-D：lr 5e-5、warmup 10%、cosine 到 1e-6、1 epoch", "IMPLEMENTATION CHOICE"],
            ["小批次", "少於 8 個樣本的批次丟掉（303 個，583 樣本）", "bug fix"],
            ["w/ ESSGNN 列", "Stage 2 的 query 頭疊在 P1 上，在 Objaverse 評、layout 關掉", "PAPER FACT §3.2"]],
        { x: 0.45, y: 1.2, w: 9.1 }, { colW: [1.2, 6.1, 1.8], fontSize: 8.5, rowH: 0.33 });
}

// 19 stage 2 results
{
  const s = base("Stage 2 初步結果");
  img(s, "report_fig5_stage2.png", { x: 0.3, y: 1.1, h: 2.55 });
  table(s, [["arm", "query 給什麼", "配方", "ProcTHOR S1 / S2-off / S2-on", "w/ ESSGNN C（text/image/pc/T+I/T+PC/I+PC/full）"],
            ["先導 2", "全部 T/I/P", "平坦 5e-4", "82.4 / 24.2 / 23.5", "10.1 / 15.2 / 49.2 / 22.2 / 56.4 / 50.5 / 58.9"],
            ["**S2-C", "只給文字（Figure 1 形式）", "5e-5 warmup cosine", "10.3 / 12.4 / 12.2", "22.5 / 37.7 / 74.3 / 62.8 / 86.1 / 80.1 / 88.2"],
            ["S2-D", "全部 T/I/P", "5e-5 warmup cosine", "82.4 / 36.8 / 32.8", "24.9 / 36.1 / 71.2 / 58.2 / 80.9 / 73.0 / 80.1"]],
        { x: 0.45, y: 3.75, w: 9.1 }, { colW: [0.7, 1.7, 1.5, 1.9, 3.3], fontSize: 8, rowH: 0.26 });
  note(s, "P1 父：34.7 / 56.9 / 86.1 / 87.6 / 99.0 / 92.7 / 99.7。論文 w/ ÷ w/o = 0.82～0.93；S2-C ÷ 父 0.65～0.88。λ 不動、S2-on ≈ S2-off：layout 在精確找資產上幫不上忙，與論文 w/ < w/o 方向一致；掉多少由配方決定。S2-C 為候選，非定案。", 4.85);
}

// ================================================================ section 05
section("05", "現況與接下來計畫", "五段排程、判斷規則、待決定事項。");

// 20 status
{
  const s = base("目前情況總結");
  const cards = [["架構與公式", "Stage 1 35 項、Stage 2 Eq. 2–8 逐項對過論文。唯一設計偏離：CLIP 文字／影像塔凍結（記憶體）", theme.secondary],
                 ["ULIP-2", "沒被動；零樣本重現論文數字（50.9 / 79.3 對 50.6 / 79.1）", theme.secondary],
                 ["Stage 1", "單模態格已貼近論文；融合格偏高的原因確定是 query 觀測與 gallery 太像（資料層），不是架構、不是訓練長短、不是 ULIP-2", theme.accent],
                 ["主線與授權", "主線 P1；Stage 2 候選 S2-C；兩者都可被接下來的實驗推翻。實驗執行人依論文證據＋實驗證據決定設定（含 lr）；只有資料集允許先不一致", theme.light]];
  cards.forEach((c, i) => card(s, { x: 0.45 + (i % 2) * 4.65, y: 1.25 + Math.floor(i / 2) * 1.95, w: 4.45, h: 1.75 }, c[0], c[1], c[2]));
}

// 21 plan
{
  const s = base("接下來計畫（一張卡，依序跑）");
  table(s, [["段", "內容", "狀態 / 預計", "判斷規則"],
            ["1", "P1 設定不變，10 代 → 25 代", "跑中，約 08:00", "full 仍高於 pc → 訓練長短不是原因，不跑 250"],
            ["2", "lr 掃描 1e-4 / 1e-3 / 3e-3，各 10 代（5e-4 已有）", "排隊，約 10:40", "dev_val 七格平均選；另報對論文距離"],
            ["3", "P8：query 三模態全換第二份觀測（另一段描述、單張視角、單邊掃描點雲），用第 2 段的 lr", "排隊，約 11:45", "full 開始低於 pc → 方向對，再掃去顏色／加雜訊"],
            ["4", "CLIP 文字／影像塔開最後幾層（論文 full encoder fine-tuning）", "之後", "先量記憶體；全開這張卡吃不下"],
            ["5", "定案 Stage 1；Stage 2 三個 arm 在新父上重跑；全量 9,600 間；iterative-prefix；ESSGNN 正文字面版；pooling / λ₀", "之後", ""]],
        { x: 0.45, y: 1.2, w: 9.1 }, { colW: [0.4, 4.4, 1.4, 2.9], fontSize: 9, rowH: 0.45 });
  card(s, { x: 0.45, y: 4.05, w: 9.1, h: 1.05 }, "待決定", "ProcTHOR 切分（80/20 或官方 10k/1k/1k）；GPT-4o 場景評審；最終評估是否解封 test；Gemma 對 GPT-4o 的標註比較需要 GPT-4o。", theme.accent);
}

// 22 summary
{
  const s = pres.addSlide(); s.background = { color: theme.primary };
  s.addText("一句話總結", { x: 0.6, y: 0.5, w: 8, h: 0.7, fontSize: 28, fontFace: ZH, color: theme.light, bold: true, margin: 0 });
  const pts = ["架構、公式、ULIP-2 都對過，沒問題。", "Stage 1 的差距在 query 觀測與 gallery 太像；架構軸和訓練長短都改不了形狀。", "P1 是目前主線、S2-C 是 Stage 2 候選；lr、epochs、query 觀測正在重跑，結果會推翻或確認。", "資料集的差異先放著，先找對方向。"];
  s.addText(pts.map((t, i) => ({ text: (i + 1) + ".  " + t, options: { fontSize: 16, fontFace: ZH, color: "FFFFFF", paraSpaceAfter: 14, breakLine: true } })), { x: 0.6, y: 1.5, w: 8.8, h: 3.4, valign: "top", margin: 0 });
  badge(s);
}

pres.writeFile({ fileName: "./output/MetaFind_report_20260904.pptx" }).then((f) => console.log("wrote", f));

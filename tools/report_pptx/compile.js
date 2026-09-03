// MetaFind 重現進度報告 2026-09-04（第二版：一頁一件事，公式用圖片，白話 + 專業）
// Numbers: docs/NOTE_20260904_*, output/look/ARMS_TABLE.md; charts: tools/probes/draw_report_charts.py,
// equations + simple charts: tools/probes/draw_report_formulas.py (all copied into ./imgs).
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";              // 10" x 5.625"
// speaker notes, one per slide in order (Kyzen: 「ppt 都沒有備忘稿 加進去啊」)
const NOTES = [
"各位好，今天報告 MetaFind 重現的進度。內容分七段：MetaFind 在做什麼、之前錯在哪、改了什麼與 Stage 1 的測試、融合分數為什麼還太高、Stage 2 做了什麼、接下來的計畫。所有數字都有出處，寫在 repo 的筆記裡。",
"先給結論，讓大家有個底。第一，模型的架構、公式、訓練方式已經逐條對過論文，backbone ULIP-2 也驗證過沒被改。第二，第一次跑出來全部 90 分以上，原因是 query 和 gallery 讀同一份資料，模型在抄答案；改掉之後單模態的分數已經接近論文。第三，融合分數還是太高，原因是 query 的點雲和文字仍然是 gallery 那個檔，這是資料層的問題，已經開始重跑。第四，Stage 2 流程已經通了，等 Stage 1 定案再重跑一次。",
"大綱七段。第一段背景；第二、三段回答有沒有照論文、之前錯在哪；第四、五段是測試與證據；第六段 Stage 2；第七段計畫。",
"第一段先把 MetaFind 講清楚，後面才知道我們在對什麼。",
"任務很單純：給幾個線索，從四萬五千多個 3D 模型裡找出「正好是那一個」。線索可以是文字描述、一張圖、或一朵點雲，三個任意組合。模型把線索算成一個向量，跟資產庫每個模型的向量比相似度，最像的排第一。R@1 就是第一名答對的比例。論文 Table 1 把七種線索組合各報 R@1 和 R@5，總共 14 格，我們的目標就是這 14 格。",
"模型是兩座塔。query 塔處理線索，gallery 塔處理資產庫。兩座塔共用同一個 ULIP-2：文字進 CLIP 文字塔、圖片進 CLIP 影像塔、點雲進 Point-BERT，各出一個 1280 維向量。然後各塔用一個 Transformer Fusion 把手上的模態合成一個向量。gallery 塔三個模態永遠齊全，訓練完就凍結，向量事先算好存起來。query 塔可以少模態，缺的模態放一個可學的 mask token，不是塞零。",
"Stage 1 是對比學習。一批 64 個資產，讓每個 query 的向量跟自己那個資產的 gallery 向量最像，跟其他 63 個不像。這就是 Eq. 5，溫度 τ 論文固定 0.5。訓練時 query 的每個模態各有 30% 機率被拿掉，這樣模型才會學到「少給線索也要找得到」。兩座塔都在學，Fusion 從零開始，Point-BERT 也一起微調。CLIP 的文字塔和影像塔我們凍結，這是唯一跟論文不同的設計，後面會講原因。",
"Stage 2 在 ProcTHOR 的房間裡微調。query 多加一個「這個房間長怎樣」的向量 e_layout，用 Eq. 6 加上去，λ 是可學權重。e_layout 由 ESSGNN 算出來，這是一個等變圖神經網路，把房間裡其他物件的位置和語意關係壓成一個向量。損失改成雙向 Eq. 7、8 取平均。gallery 塔和 ULIP-2 全部凍結。每個 batch 有 30% 機率不給 layout，模型要能沒有房間資訊也找得到。",
"這是我們要對的成績單。請看形狀，不只看高低：只給點雲 75 分是最高的，再加文字或圖片，分數反而掉到 44 到 52。這個形狀在說一件事：論文 query 的文字和圖片，跟資產庫裡那一份不是同一份，所以弱線索會把強線索拉下來。如果是同一份，加線索只會更準。論文沒寫 query 是怎麼取得的，也沒寫學習率、幾代、batch，這些都得靠實驗補。",
"我們的資料和論文有幾個差異，你已經裁定先放著。資產 45,692 對論文的約 48,000；每個資產 12 張圖對論文 11 張，量過只用 11 張形狀不變；文字標註用 gemma 產生論文 Figure 2 的 13 個欄位，論文用 GPT-4o；切分我們多切一塊驗證集，測試集封存不碰。評估用 C 和 D 兩個協定，所有數字都走同一個評估器。",
"第二段回答三個問題：Stage 1 有沒有照論文的規則、哪些在學、一步怎麼跑。最後講兩個沒對齊的怎麼解。",
"逐條對。論文有寫的規則：共用 ULIP-2、兩塔都訓、30% 獨立遮罩、mask token 不用零、gallery 三模態齊全、Eq. 5 單向 τ 0.5、Transformer Fusion，全部照做。兩個沒對齊：論文方法段沒有明寫編碼器要微調，是 §3.4 的消融「只訓 fuser 比較差」推出主模型有微調，哪幾層沒寫；我們 Point-BERT 有訓、CLIP 兩座塔沒訓；學習率、代數、batch、優化器論文沒寫，由實驗定。",
"左邊是在學的和不動的：Point-BERT、兩個 Fusion、三個 mask token 在學；CLIP 兩座塔不動，向量事先算好。右邊是一步的六個動作：抽 64 個資產；gallery 側三個模態算向量、L2、進 Fusion；query 側文字同一句、圖片只挑一張、點雲同一個檔進同一顆 Point-BERT，每個模態 30% 換 mask token，L2，進 Fusion；64 乘 64 算 cosine 除以 0.5 套 Eq. 5；梯度回傳，AdamW 更新；一代約 500 步，跑 10 代。每代結束在驗證集選最好那代。",
"兩個沒對齊的怎麼解。CLIP 沒微調的問題在算力：影像塔一張圖前向約 1 TFLOP，每步要重算 768 張，這張卡一代一小時以上。方案 A 現在做：只開最後的投影層和文字塔最後一個 block，向量照舊快取，記成部分微調。方案 B 要另一張 80 GB 的卡，要你決定。學習率等論文沒寫的，規則寫死只看驗證集：優化器照 ULIP 官方、batch 64、學習率掃四個值、代數 10 對 25 比。",
"第三段：之前數據為什麼差那麼多，錯在哪。",
"第一次跑出來七種線索全部 90 分以上，三個都給是 100。這是壞消息。只給一句「木頭椅子」就 58 分，在三萬六千個模型裡不可能認出那一張椅子。這代表模型不是在認東西，是在認出「同一份資料」。",
"三個錯誤。主因：第 3 步 query 讀的是第 2 步同一個 dict、同一批張量，模型學到照抄，七格全 90 以上，證據是 cosine 0.9989。第二：文字字串把 160 字描述放最前面，只給文字 58 分，換填表後掉到 8。第三：三個模態沒正規化就進 Fusion，點雲向量長度 139 壓過文字 37、圖片 40。修掉以後 P1 只給文字 11.6、只給點雲 66.6，論文 13.8、75.1。還沒修好的一塊：query 的點雲和文字還是跟 gallery 同一個檔，含點雲的組合都還 95 以上，正在跑 P8。",
"主因在這裡。舊程式把同一個 dict、同一批張量同時餵給兩座塔。Eq. 5 在這種情況下有一個偷懶解：兩座塔輸出一模一樣，loss 就最小。用考試比喻，題目卷和答案卷是同一張紙。證據：實測 query 向量和 gallery 向量的 cosine 是 0.9989，三個都給 99.98。為什麼確定論文不是這樣？因為論文三個都給只有 51.7，如果論文也讀同一份，會跟我們一樣接近 100。所以論文的 query 一定是另一份觀測。",
"其他六個問題。文字太長，論文的文字輸入是短填表；點雲向量長度 139 壓過文字 37 和圖片 40，要先正規化；一個分析工具把 gallery 的點雲餵成全零，結論全部撤回；Stage 2 建模型漏帶兩個設定；Stage 2 批次尾巴只剩 1 個樣本 loss 變 0；還有評估各處不一致。每項都修了，也都有測試釘住。",
"第四段：改了什麼，以及 Stage 1 做了哪些測試。",
"每一個改變都對應論文哪一句。query 改讀第二份觀測，依據是 Table 1 的 51.7；文字改填表，依據是 Figure 2 的欄位；進 Fusion 前正規化是我們量到問題後的選擇；評估器統一；Stage 2 的兩個 bug 修正；Stage 2 配方論文只說 fine-tuning，我們選 5e-5 加 warmup 和 cosine。",
"測試設計的原則：一次只換一個東西，看成績單的形狀會不會變。八個版本用同一份資料、同一組訓練設定、同一個評估器。P1 是三個修正一起上；P3 到 P7 各在 P1 上換一件事；P5 是三個模態都換第二份觀測。共同設定在下面：AdamW、學習率 5e-4、10 代、每代在驗證集選最好那代。",
"這是目前的數據。level 是整體比論文高多少，shape 是扣掉整體高低後形狀像不像。只給文字 11.6 對論文 13.8、只給點雲 66.6 對 75.1，已經接近。歪掉的全是含點雲的組合：論文加線索會掉到 44，我們一路升到 98。",
"同一份數據畫成圖。黑線是論文。所有彩色線在點雲那格跟論文交叉之後就一路往上，沒有一條往下。",
"重點結論：七個新版本的 shape 全在 0.40 到 0.44。Fusion 一份或兩份、圖片平均或逐張、正規化開或關、視角固定或隨機，形狀都一樣。如果問題在架構，換架構形狀就會變。它沒變，所以問題不在架構，在資料。第五段證明這件事。",
"為什麼選 P1：它最乾淨，三個修正各有論文依據；單模態最接近論文；其他版本沒有更好；而且它可以被推翻，學習率、代數、query 觀測都在重跑。",
"第五段：融合分數為什麼還太高。這一段回答你上午的問題：是不是 ULIP-2 被改了？",
"先排除 ULIP-2。程式碼 17 個檔案跟上游逐 byte 相同，補丁只做相容，沒改數學。用官方釋出的權重、照官方的評估程式，在我們的資料上做零樣本分類，top-1 50.9、top-5 79.3，論文是 50.6、79.1。所以 backbone 就是論文的 ULIP-2，分數高不是這裡來的。",
"ULIP-2 說的「拉近三個模態」是什麼意思？實測：文字向量跟自己的點雲 0.29、跟別人平均 0.07；圖片 0.46 對 0.11。拉近的意思是自己的比別人的近，不是三個點疊成一個點。這也解釋為什麼只給文字很低：0.29 對 0.07 差距很小，總有某個很像的別人比自己更近。文字是分不清，不是指向別人。",
"這是最關鍵的一頁。query 的點雲跟 gallery 的點雲是同一個檔，相似度 1.00，最像的別人只有 0.59。加文字，query 離自己掉到 0.80，但離別人也掉到 0.53。再加圖片，0.76 對 0.52。差距一直在，名次翻不了，所以只會越加越高。下面的式子是為什麼：文字要翻轉名次，它偏向別人的程度必須超過點雲本身的差距，這不可能發生。而且不用 Fusion、不訓練、單純平均也一樣越加越高。所以不是 Fusion 學出篩選線索，是資料讓它只能往上。",
"那為什麼 P1 只給點雲是 66.6 不是 100？因為 gallery 存的是三個模態融合過的向量，query 只給點雲時是點雲加兩個 mask token 進 Fusion，走的路不一樣，所以掉分。三個都給時走同一條路，而且點雲和文字是同一個檔，幾乎自己對自己，回到 98。",
"用實驗確認。把 query 的點雲換成同一個物件的不同看法：去顏色、加雜訊、只留一半，其他不動，P1 直接重評。點雲分數掉很快，但三個都給永遠高於只給點雲。兩個發現：P1 的點雲塔很脆，稍微動一下就從 66 掉到個位數；而且光弄壞點雲做不出論文的形狀，文字和圖片也要是不同觀測。這就是接下來的 P8 實驗。",
"訓練夠不夠？論文沒寫幾代，250 是 ULIP 官方程式碼的預設值。P1 從第 0 代起三個都給就高於只給點雲，越練七格一起往上；舊版本跑 44 代七格全 90 以上。所以不是訓練長短的問題。正在用 25 代和學習率掃描驗證這件事。",
"第六段：Stage 2。",
"Stage 2 的設定。先用 ProcTHOR 前 1,500 間房驗證流程，將近十萬個樣本，資產庫 1,439 個。gallery 塔和 ULIP-2 凍結，只訓 query 的 Fusion 和 ESSGNN，這是論文 §2.6。λ 初值是 Fusion 向量長度中位數的十分之一。場景 dropout 30%，雙向損失，都照論文。學習率論文沒給，我們選 5e-5 加 warmup 和 cosine。Table 1 的 w/ ESSGNN 列是把 Stage 2 訓好的 query 頭疊回 P1，在 Objaverse 上評、layout 關掉，這是論文 §3.2 的做法。",
"結果。開 layout 和關 layout 幾乎一樣，λ 三次都不動：layout 分支在精確找出同一個資產這件事上幫不上忙，這跟論文 w/ 比 w/o 低的方向一致。掉多少由學習率決定：平坦 5e-4 把頭訓壞，5e-5 加 warmup 和 cosine 把損傷縮到論文的量級。S2-C 只給文字，最像論文 Figure 1 的 query，數字最接近論文的 w/ 列，是候選，不是定案。等 Stage 1 定案後三個版本要重跑。",
"最後一段：計畫。",
"目前情況四句話：架構與公式對過了，唯一偏離是 CLIP 兩座塔凍結，記憶體不夠；ULIP-2 沒被動；Stage 1 融合分數偏高的原因確定是資料層；主線 P1、Stage 2 候選 S2-C，都可以被實驗推翻。設定由論文證據和實驗證據決定，只有資料集允許先不一致。",
"排程五段，一張卡一個接一個跑。第一段 P1 跑 25 代，看訓練長短；第二段學習率掃描；第三段 P8，query 三個模態全換第二份觀測，這是證據最強的方向；第四段 CLIP 開最後幾層；第五段定案 Stage 1、Stage 2 重跑。每段有判斷規則。要你決定的三件事在下面。",
"一句話總結：模型跟論文一樣了；第一次分數太高是拿答案對答案；融合分數還太高是 query 跟 gallery 是同一個檔，正在重跑；P1 和 S2-C 是目前主線，實驗會確認或推翻。謝謝。"
];
{ const _add = pres.addSlide.bind(pres); let k = 0; pres.addSlide = () => { const sl = _add(); const n = NOTES[k++]; if (n) sl.addNotes(n); return sl; }; }

const theme = { primary: "264653", secondary: "2a9d8f", accent: "e76f51", light: "e9c46a", bg: "fbfaf6" };
const ZH = "Microsoft YaHei", EN = "Arial";
const INK = "1f2a30", GREY = "5c6b73", CARD = "ffffff", LINE = "d9d4c7", PALE = "f3efe6";

const IMG_DIMS = {"report_fig2_level_shape.png": [1170, 624], "report_fig4_query_pc_observation.png": [1560, 780], "report_fig1_stage1_arms_D.png": [1560, 780], "report_fig6_ulip2_zero_shot.png": [845, 546], "ulip2_pull_explainer.png": [1820, 1430], "report_fig3_P1_epochs.png": [1300, 715], "report_fig5_stage2.png": [1820, 715],
  "eq5.png": [2196, 320], "eq6.png": [1516, 200], "eq78.png": [2400, 320], "eq_trivial.png": [2400, 200], "eq_mean.png": [2301, 110], "fig_own_vs_other.png": [1280, 672], "fig_shape_paper_vs_p1.png": [1280, 672]};

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
  s.addText(title, { x: 0.45, y: 0.2, w: 8.8, h: 0.6, fontSize: 24, fontFace: ZH, color: theme.primary, bold: true, margin: 0, fit: "shrink" });
  if (subtitle) s.addText(subtitle, { x: 0.45, y: 0.76, w: 9.1, h: 0.3, fontSize: 11, fontFace: ZH, color: GREY, margin: 0, fit: "shrink" });
  badge(s);
  return s;
}
// the one sentence of the slide
function headline(s, text, y, color) {
  s.addShape(pres.shapes.RECTANGLE, { x: 0.45, y: y || 1.12, w: 9.1, h: 0.62, fill: { color: color || theme.primary }, line: { color: color || theme.primary } });
  s.addText(text, { x: 0.6, y: y || 1.12, w: 8.8, h: 0.62, fontSize: 15, fontFace: ZH, color: "FFFFFF", bold: true, valign: "middle", margin: 0, fit: "shrink" });
}
function bullets(s, items, box, size) {
  size = size || 12;
  const runs = items.map((it) => {
    const lvl = Array.isArray(it) ? it[0] : 0, txt = Array.isArray(it) ? it[1] : it;
    const bold = txt.startsWith("**"); const t = txt.replace(/\*\*/g, "");
    return { text: t, options: { bullet: lvl === 0 ? { indent: 12 } : { indent: 12, code: "2013" }, indentLevel: lvl, fontSize: lvl ? size - 1 : size, fontFace: ZH, color: lvl ? GREY : INK, bold: bold, paraSpaceAfter: 6, breakLine: true } };
  });
  s.addText(runs, Object.assign({ fontFace: ZH, valign: "top", margin: 2, fit: "shrink" }, box));
}
function para(s, text, box, size, color) {
  s.addText(text, Object.assign({ fontSize: size || 12, fontFace: ZH, color: color || INK, valign: "top", margin: 2, fit: "shrink" }, box));
}
function table(s, rows, box, opts) {
  opts = opts || {};
  const fs = opts.fontSize || 9;
  const data = rows.map((r, i) => r.map((c) => {
    const txt = String(c); const bold = txt.startsWith("**"); const t = txt.replace(/\*\*/g, "");
    return { text: t, options: { fontSize: fs, fontFace: ZH, color: i === 0 ? "FFFFFF" : INK, bold: i === 0 || bold, fill: { color: i === 0 ? theme.primary : (i % 2 ? "FFFFFF" : PALE) }, valign: "middle", margin: 0.03 } };
  }));
  s.addTable(data, Object.assign({ border: { type: "solid", pt: 0.5, color: LINE }, colW: opts.colW, rowH: opts.rowH || 0.26, autoPage: false }, box));
}
function card(s, box, title, body, color, size) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, Object.assign({ fill: { color: CARD }, line: { color: LINE, pt: 0.75 }, rectRadius: 0.08 }, box));
  s.addShape(pres.shapes.RECTANGLE, { x: box.x, y: box.y, w: 0.08, h: box.h, fill: { color: color || theme.secondary } });
  s.addText(title, { x: box.x + 0.18, y: box.y + 0.08, w: box.w - 0.3, h: 0.32, fontSize: 12, fontFace: ZH, color: theme.primary, bold: true, margin: 0 });
  s.addText(body, { x: box.x + 0.18, y: box.y + 0.42, w: box.w - 0.3, h: box.h - 0.5, fontSize: size || 10.5, fontFace: ZH, color: INK, valign: "top", margin: 0, fit: "shrink" });
}
function section(num, title, intro) {
  const s = pres.addSlide();
  s.background = { color: theme.primary };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 3.2, h: 5.625, fill: { color: theme.secondary } });
  s.addText(num, { x: 0.4, y: 1.6, w: 2.5, h: 1.6, fontSize: 88, fontFace: EN, color: "FFFFFF", bold: true, margin: 0 });
  s.addText(title, { x: 3.7, y: 1.7, w: 5.8, h: 0.9, fontSize: 30, fontFace: ZH, color: "FFFFFF", bold: true, margin: 0, fit: "shrink" });
  if (intro) s.addText(intro, { x: 3.7, y: 2.7, w: 5.8, h: 1.6, fontSize: 14, fontFace: ZH, color: "e9e4d8", margin: 0, valign: "top", fit: "shrink" });
  badge(s);
}
function img(s, file, box) {
  const [pw, ph] = IMG_DIMS[file]; const ar = pw / ph;
  let w = box.w, h = box.h;
  if (!w) w = h * ar;
  if (!h) h = w / ar;
  const maxW = box.maxW || (9.7 - box.x);
  if (w > maxW) { w = maxW; h = w / ar; }
  s.addImage({ path: "./imgs/" + file, x: box.x, y: box.y, w: w, h: h });
}
function boxflow(s, items, x, y, w, h, gap) {
  // vertical boxes with arrows between them
  items.forEach((b, i) => {
    const yy = y + i * (h + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x, y: yy, w: w, h: h, fill: { color: b[2] }, line: { color: b[2] }, rectRadius: 0.06 });
    s.addText([{ text: b[0], options: { bold: true, fontSize: 11, breakLine: true } }, { text: b[1], options: { fontSize: 9 } }], { x: x + 0.12, y: yy, w: w - 0.24, h: h, fontFace: ZH, color: "FFFFFF", valign: "middle", margin: 0 });
    if (i < items.length - 1) s.addText("▼", { x: x, y: yy + h - 0.02, w: w, h: gap + 0.04, fontSize: 9, color: GREY, align: "center", valign: "middle", margin: 0 });
  });
}

// ================================================================ cover
{
  const s = pres.addSlide();
  s.background = { color: theme.primary };
  s.addShape(pres.shapes.RECTANGLE, { x: 6.3, y: 0, w: 3.7, h: 5.625, fill: { color: theme.secondary } });
  s.addShape(pres.shapes.OVAL, { x: 6.9, y: 1.3, w: 2.6, h: 2.6, fill: { color: theme.light }, line: { color: theme.light } });
  s.addText("MetaFind", { x: 6.9, y: 2.1, w: 2.6, h: 1.0, fontSize: 30, fontFace: EN, color: theme.primary, bold: true, align: "center", valign: "middle", margin: 0 });
  s.addText("MetaFind 重現進度報告", { x: 0.6, y: 1.5, w: 5.6, h: 1.0, fontSize: 36, fontFace: ZH, color: "FFFFFF", bold: true, margin: 0, fit: "shrink" });
  s.addText("Stage 1 做了什麼測試、目前數據、之前錯在哪與原因、做了什麼改變、Stage 2 做了什麼、接下來計畫", { x: 0.6, y: 2.6, w: 5.4, h: 1.1, fontSize: 14, fontFace: ZH, color: "e9e4d8", margin: 0, valign: "top" });
  s.addText("2026-09-04", { x: 0.6, y: 4.5, w: 3, h: 0.4, fontSize: 12, fontFace: EN, color: theme.light, margin: 0 });
}

// ================================================================ one-page conclusion
{
  const s = base("先講結論（一頁）");
  const cards = [["1. 模型跟論文一樣了", "架構、公式、訓練方式逐條對過論文。ULIP-2 backbone 也驗證過沒被動（零樣本分類重現論文數字）。", theme.secondary],
                 ["2. 第一次的分數太高，原因找到了", "之前 query 和 gallery 讀的是同一份資料，模型等於「拿答案對答案」，所以全部 90 分以上。改掉之後單模態分數已經接近論文。", theme.accent],
                 ["3. 還差一塊：融合分數還是太高", "原因也找到了：query 的點雲和文字仍然跟 gallery 是同一個檔。這是資料層的問題，不是架構。已經開始用「不同觀測」重跑。", theme.accent],
                 ["4. Stage 2 已經跑起來", "ESSGNN 佈局分支、Table 1 的 w/ ESSGNN 列都能產出數字；候選版本 S2-C。等 Stage 1 定案後重跑一次。", theme.light]];
  cards.forEach((c, i) => card(s, { x: 0.45 + (i % 2) * 4.65, y: 1.2 + Math.floor(i / 2) * 1.95, w: 4.45, h: 1.8 }, c[0], c[1], c[2], 11));
}

// ================================================================ TOC
{
  const s = pres.addSlide(); s.background = { color: theme.bg };
  s.addText("大綱", { x: 0.6, y: 0.4, w: 5, h: 0.7, fontSize: 30, fontFace: ZH, color: theme.primary, bold: true, margin: 0 });
  const items = [["01", "MetaFind 在做什麼", "任務、模型、怎麼訓練、論文的成績單"],
                 ["02", "Stage 1 有沒有照論文", "逐條對規則；哪些在學；一步怎麼跑；沒對齊的怎麼解"],
                 ["03", "之前數據為什麼差那麼多", "三個錯誤、效果、證據；修掉後的數字"],
                 ["04", "改了什麼、做了哪些測試", "八個版本、目前數據、為什麼選 P1"],
                 ["05", "融合分數為什麼還太高", "ULIP-2 沒被動；query 跟 gallery 太像；實驗證明"],
                 ["06", "Stage 2 做了什麼", "佈局分支、四次跑的結果、候選 S2-C"],
                 ["07", "接下來計畫", "排程、判斷規則、要你決定的事"]];
  items.forEach((it, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 4.7, y = 1.25 + row * 1.02;
    s.addText(it[0], { x: x, y: y, w: 0.9, h: 0.8, fontSize: 30, fontFace: EN, color: theme.accent, bold: true, margin: 0 });
    s.addText(it[1], { x: x + 0.95, y: y, w: 3.6, h: 0.42, fontSize: 15, fontFace: ZH, color: theme.primary, bold: true, margin: 0 });
    s.addText(it[2], { x: x + 0.95, y: y + 0.4, w: 3.6, h: 0.5, fontSize: 9.5, fontFace: ZH, color: GREY, margin: 0, valign: "top" });
  });
  badge(s);
}

// ================================================================ 01
section("01", "MetaFind 在做什麼", "任務是什麼、模型長什麼樣、怎麼訓練、論文的成績單（Table 1）長什麼樣。");

{
  const s = base("任務：用任何線索，從資產庫找出「那一個」3D 模型");
  headline(s, "給文字、圖片、點雲的任意組合，在 45,692 個 3D 模型裡找出正確的那一個。");
  boxflow(s, [["Query（線索）", "文字「木頭椅子…」、一張照片、一朵點雲，可以只給其中幾個", theme.secondary],
              ["模型算出一個向量", "把線索變成一串數字", theme.primary],
              ["跟資產庫每個模型的向量比相似度", "資產庫的向量事先算好存著", theme.primary],
              ["排名", "最像的排第一", theme.accent]], 0.45, 1.95, 4.2, 0.65, 0.18);
  card(s, { x: 5.0, y: 1.95, w: 4.55, h: 1.45 }, "指標 R@1 / R@5", "R@1 = 排第一名的就是正確那一個，佔全部查詢的比例。\nR@5 = 正確的在前五名內的比例。", theme.secondary, 11);
  card(s, { x: 5.0, y: 3.55, w: 4.55, h: 1.6 }, "論文 Table 1 有七種線索組合", "只給文字、只給圖片、只給點雲、文字+圖片、文字+點雲、圖片+點雲、三個都給。\n每種各報 R@1 和 R@5，共 14 格。我們的目標就是這 14 格。", theme.accent, 11);
}

{
  const s = base("模型長什麼樣：兩座塔，共用一個 ULIP-2", "論文 §2.2、§2.4（PAPER FACT）");
  headline(s, "query 塔負責「線索」，gallery 塔負責「資產庫」。兩座塔用同一個 ULIP-2 把每個模態變成向量，再各自用一個 Fusion 合成一個向量。");
  boxflow(s, [["ULIP-2（上游現成模型）", "文字 → CLIP 文字塔；圖片 → CLIP 影像塔；點雲 → Point-BERT。各出一個 1280 維向量", theme.primary],
              ["Fusion（Transformer）", "把手上有的模態向量合成一個。缺的模態放一個可學的「mask token」，不是塞零", theme.primary],
              ["相似度", "cosine，除以溫度 τ = 0.5", theme.accent]], 0.45, 1.95, 4.4, 0.85, 0.2);
  card(s, { x: 5.15, y: 1.95, w: 4.4, h: 1.45 }, "gallery 塔", "三個模態永遠齊全。訓練完就凍結，45,692 個向量事先算好存起來。", theme.secondary, 11);
  card(s, { x: 5.15, y: 3.55, w: 4.4, h: 1.6 }, "query 塔", "可以少模態。訓練時故意隨機拿掉模態（下一頁），所以少給也找得到。Stage 2 再多加一個「房間佈局」向量。", theme.accent, 11);
}

{
  const s = base("怎麼訓練 Stage 1：對比學習", "論文 §2.6 Stage 1、Eq. 5（PAPER FACT）");
  headline(s, "一批 64 個資產。讓每個 query 向量跟「自己那個資產」的 gallery 向量最像，跟其他 63 個不像。");
  img(s, "eq5.png", { x: 0.6, y: 1.95, w: 8.8 });
  card(s, { x: 0.45, y: 3.35, w: 2.95, h: 1.75 }, "sim 是什麼", "cosine 相似度：兩個向量夾角越小越像。τ = 0.5 是溫度，論文固定，全實驗一樣。", theme.secondary, 10.5);
  card(s, { x: 3.55, y: 3.35, w: 2.95, h: 1.75 }, "30% 遮罩", "訓練時 query 的每個模態各有 30% 機率被拿掉，換成 mask token。這樣模型學會「少給也要找得到」。", theme.secondary, 10.5);
  card(s, { x: 6.65, y: 3.35, w: 2.9, h: 1.75 }, "兩座塔都訓", "query 塔和 gallery 塔都在學。Fusion 從零開始學；Point-BERT 也一起微調；CLIP 文字／影像塔我們凍結（後面會講這是唯一的偏離）。", theme.accent, 10.5);
}

{
  const s = base("怎麼訓練 Stage 2：加房間佈局", "論文 §2.6 Stage 2、Eq. 6、7、8（PAPER FACT）");
  headline(s, "在 ProcTHOR 房間裡做微調：query 多加一個「這個房間長怎樣」的向量，gallery 塔和 ULIP-2 全部凍結。");
  img(s, "eq6.png", { x: 0.6, y: 1.95, w: 5.5 });
  img(s, "eq78.png", { x: 0.6, y: 2.85, w: 8.8 });
  card(s, { x: 0.45, y: 4.1, w: 4.45, h: 1.05 }, "e_layout 從哪來", "ESSGNN：一個等變圖神經網路，把房間裡其他物件的位置和語意關係算成一個向量。λ 是可學的權重。", theme.secondary, 10.5);
  card(s, { x: 5.1, y: 4.1, w: 4.45, h: 1.05 }, "30% 場景 dropout", "每個 batch 有 30% 機率不給 layout，模型要能沒有房間資訊也找得到。", theme.accent, 10.5);
}

{
  const s = base("論文的成績單長什麼樣：這是我們要對的目標", "論文 Table 1，MetaFind w/o ESSGNN 列（R@1）");
  headline(s, "重點不只是數字高低，是「形狀」：只給點雲 75 分最高；再加文字或圖片，分數反而掉到 44～52。");
  table(s, [["線索", "只給文字", "只給圖片", "只給點雲", "文字+圖片", "文字+點雲", "圖片+點雲", "三個都給"],
            ["**論文 R@1", "13.8", "11.7", "**75.1", "17.2", "**44.5", "**45.8", "**51.7"],
            ["論文 R@5", "23.1", "19.2", "78.0", "21.8", "71.3", "73.1", "76.5"]],
        { x: 0.45, y: 1.95, w: 9.1 }, { colW: [1.3, 1.1, 1.1, 1.1, 1.15, 1.15, 1.15, 1.15], fontSize: 11, rowH: 0.34 });
  card(s, { x: 0.45, y: 3.2, w: 4.45, h: 1.9 }, "這個形狀在說什麼", "加了較弱的線索（文字、圖片）會把強線索（點雲）拉低。這只有在「query 的文字、圖片跟資產庫裡那一份不是同一份」時才會發生。如果是同一份，加線索只會更準。", theme.accent, 11);
  card(s, { x: 5.1, y: 3.2, w: 4.45, h: 1.9 }, "論文沒寫的", "query 的文字、圖片、點雲是怎麼取得的；學習率、幾代、batch、優化器；Fusion 幾層。這些都得靠實驗補。", theme.secondary, 11);
}

{
  const s = base("我們的資料，跟論文差在哪", "你已裁定：資料集允許先不一致，其他訓練設計要跟論文一樣");
  table(s, [["項目", "我們", "論文", "差異"],
            ["資產數", "45,692（官方 LVIS 清單 46,052，扣掉點雲／渲染／標註缺的）", "約 48,000", "做不到 48K"],
            ["每個資產的圖片", "12 個視角（OpenShape 三圈 × 4）", "11 個正交視角", "量過只用 11 張，成績單形狀不變"],
            ["點雲", "10,000 點，帶顏色，ULIP 的正規化", "同 ULIP-2", "一樣"],
            ["文字標註", "gemma-4-12B-it 產生 13 個欄位（論文 Figure 2 的欄位）", "GPT-4o", "模型不同"],
            ["切分", "70 / 10 / 20：訓練 31,985、驗證 4,569、測試 9,138（封存不碰）。論文 80/20 不變：80% = 70% 訓練 + 10% 驗證", "80 / 20", "論文 80/20 不變；80% 內切 10% 當驗證，測試 20% 封存"],
            ["評估", "C：4,569 對 4,569（選 checkpoint 用）；D：4,569 對 36,554（主要比對表）", "gallery 大小沒寫", ""]],
        { x: 0.45, y: 1.2, w: 9.1 }, { colW: [1.3, 4.3, 1.6, 1.9], fontSize: 10, rowH: 0.42 });
  para(s, "評估器：float64 cosine，同分算模型輸。所有數字都用同一個評估器算，不然比不了。", { x: 0.45, y: 4.3, w: 9.1, h: 0.5 }, 11, GREY);
}

// ================================================================ 02
section("02", "Stage 1 有沒有照論文？哪些在學？一步怎麼跑？", "逐條對論文規則；在學的和不動的；一步的六個動作；兩個沒對齊的怎麼解。");

{
  const s = base("Stage 1 有沒有照論文規則？逐條對");
  table(s, [["論文規則（出處）", "我們", "符合"],
            ["兩座塔共用同一個 ULIP-2（Figure 1「Shared」）", "一顆 Point-BERT、一組 CLIP，兩塔共用", "符合"],
            ["query 塔、gallery 塔都要訓（Stage 1 段）", "兩邊的 Fusion 都在學", "符合"],
            ["query 每個模態各 30% 機率獨立遮掉（Stage 1 段）", "30%，各自獨立丟硬幣", "符合"],
            ["遮掉的用 masked embedding，不用零（Stage 1 段）", "可學的 mask token", "符合"],
            ["gallery 三模態齊全（§2.4）", "永遠齊全", "符合"],
            ["損失 Eq. 5，單向，τ = 0.5（§3.1）", "單向 InfoNCE，τ 固定 0.5", "符合"],
            ["Fusion 用 Transformer（§3.4）", "Transformer", "符合"],
            ["主模型有微調編碼器，不是只訓 Fusion（§3.4 消融推得，Table 3「Train fuser only」較差；哪幾層沒寫）", "Point-BERT 有訓；CLIP 兩座塔沒訓", "部分（推論，非明文）"],
            ["學習率、代數、batch、優化器", "論文沒寫", "由實驗定"]],
        { x: 0.45, y: 1.2, w: 9.1 }, { colW: [4.2, 3.4, 1.5], fontSize: 10, rowH: 0.36 });
  para(s, "結論：論文有寫的規則全部照做。兩個沒對齊的，下一頁講怎麼解。", { x: 0.45, y: 4.95, w: 9.1, h: 0.4 }, 11, theme.accent);
}

{
  const s = base("哪些在學、哪些不動；一步怎麼跑（P1）");
  card(s, { x: 0.45, y: 1.2, w: 2.9, h: 1.55 }, "在學的", "Point-BERT（點雲編碼器）\nquery 的 Fusion\ngallery 的 Fusion\n三個 mask token", theme.accent, 11);
  card(s, { x: 0.45, y: 2.9, w: 2.9, h: 1.1 }, "不動的", "CLIP 文字塔\nCLIP 影像塔（向量事先算好存著）", theme.secondary, 11);
  card(s, { x: 0.45, y: 4.15, w: 2.9, h: 1.0 }, "每代結束", "在驗證集算七格 R@1 平均，最好的那代留下", theme.light, 10.5);
  const steps = [["1", "抽 64 個資產", "從訓練集抽一批"],
                 ["2", "gallery 側算向量", "文字向量（事先算好）、圖片 12 張平均（事先算好）、點雲檔進 Point-BERT。三個各自 L2，進 gallery Fusion → e_gallery"],
                 ["3", "query 側算向量", "文字同一句、圖片只挑一張、點雲同一個檔進同一顆 Point-BERT。每個模態丟硬幣 30% 換 mask token。L2，進 query Fusion → e_query"],
                 ["4", "算損失", "64 × 64 的 cosine，除以 0.5，套 Eq. 5：自己那格要最大"],
                 ["5", "更新", "梯度回到 Point-BERT、兩個 Fusion、mask token；AdamW 更新"],
                 ["6", "重複", "一代約 500 步；10 代"]];
  steps.forEach((st, i) => {
    const y = 1.2 + i * 0.66;
    s.addShape(pres.shapes.OVAL, { x: 3.6, y: y + 0.08, w: 0.42, h: 0.42, fill: { color: theme.primary } });
    s.addText(st[0], { x: 3.6, y: y + 0.08, w: 0.42, h: 0.42, fontSize: 13, fontFace: EN, color: "FFFFFF", bold: true, align: "center", valign: "middle", margin: 0 });
    s.addText([{ text: st[1], options: { bold: true, fontSize: 11.5, color: theme.primary, breakLine: true } }, { text: st[2], options: { fontSize: 9.5, color: INK } }],
              { x: 4.15, y: y, w: 5.45, h: 0.62, fontFace: ZH, valign: "middle", margin: 0, fit: "shrink" });
  });
}

{
  const s = base("兩個沒對齊的，怎麼解");
  card(s, { x: 0.45, y: 1.2, w: 4.45, h: 3.95 }, "CLIP 兩座塔沒微調（論文沒明說要，§3.4 消融暗示有）",
       "問題不在參數量，在算力：影像塔 ViT-bigG 一張圖前向約 1 TFLOP；要訓它，每步 64 個資產 × 12 張 gallery 圖 = 768 張要重算，這張卡一代要一小時以上。現在是圖片向量事先算好存著才跑得動。\n\n方案 A（現在做）：前面的層鎖住、向量照舊快取，只打開最後的投影層（文字、影像各一個 1664→1280 矩陣）和文字塔最後一個 block。影像再往前一層要存每張圖 257 個 token，470 GB，存不下。記成 DEVIATION（部分微調）。\n\n方案 B（要另一張卡）：真的開最後 N 層，圖片每步重算。需要 80 GB 的卡或第二張卡。要你決定。", theme.accent, 10);
  card(s, { x: 5.1, y: 1.2, w: 4.45, h: 3.95 }, "學習率、代數、batch、優化器（論文沒寫）",
       "規則寫死，只看驗證集，不看論文數字、不看測試集：\n\n優化器、warmup、weight decay：照 ULIP 官方程式碼（AdamW、warmup 1 代、cosine、wd 0.1）。已採用。\n\nbatch：64（ULIP 官方單卡 64）。已採用。\n\n學習率：掃 1e-4 / 5e-4 / 1e-3 / 3e-3，各 10 代，驗證集七格平均最高者勝。排隊中。\n\n代數：10 對 25 比，成績單形狀不變就停在 10，會變再加。25 代跑中。\n\n選完把「離論文多遠」另外報，兩個分開。", theme.secondary, 10);
}

// ================================================================ 03
section("03", "之前數據為什麼差那麼多？錯在哪", "第一次全部 90 分以上。三個錯誤，每個有效果和證據；修掉後單模態接近論文；還剩一塊。");

{
  const s = base("第一次的結果：太好了，好到不合理");
  headline(s, "舊做法（pilot10b）七種線索全部 90 分以上，三個都給是 100。論文最高才 75。");
  table(s, [["線索", "只給文字", "只給圖片", "只給點雲", "文字+圖片", "文字+點雲", "圖片+點雲", "三個都給"],
            ["**舊做法 R@1", "58.0", "84.6", "78.8", "96.5", "99.6", "94.1", "100.0"],
            ["論文 R@1", "13.8", "11.7", "75.1", "17.2", "44.5", "45.8", "51.7"]],
        { x: 0.45, y: 1.95, w: 9.1 }, { colW: [1.3, 1.1, 1.1, 1.1, 1.15, 1.15, 1.15, 1.15], fontSize: 11, rowH: 0.34 });
  card(s, { x: 0.45, y: 3.2, w: 9.1, h: 1.9 }, "為什麼「太好」是壞消息", "只給文字就 58 分：一句「木頭椅子」不可能在 36,554 個模型裡認出「那一張」椅子。這代表模型不是在認東西，是在「認出同一份資料」。下一頁講原因。", theme.accent, 12);
}

{
  const s = base("錯在哪：三個錯誤、效果、證據");
  table(s, [["錯誤", "效果", "證據"],
            ["主因：第 3 步的 query 讀的是第 2 步同一個 dict、同一批張量。連遮罩前的向量都是同一份", "模型學到「照抄」：兩塔輸出一樣，七格全 90 以上，三個都給 100", "實測 query 對 gallery cosine 0.9989"],
            ["文字字串把 160 字自由描述放最前面", "只給文字 58 分。論文的文字輸入是短填表，應該只有十幾分", "換填表後 58 掉到 8"],
            ["三個模態沒正規化就進 Fusion（論文沒寫；我們的選擇）", "點雲向量長度 139、文字 37、圖片 40，點雲壓過其他兩個。修了之後形狀不變（P1 對 P7），是衛生問題不是主因", "量進 Fusion 前的向量長度"]],
        { x: 0.45, y: 1.2, w: 9.1 }, { colW: [3.9, 3.2, 2.0], fontSize: 10, rowH: 0.62 });
  card(s, { x: 0.45, y: 3.85, w: 4.45, h: 1.3 }, "修掉以後（P1）", "只給文字 11.6、只給點雲 66.6。論文 13.8、75.1。接近了。", theme.secondary, 11);
  card(s, { x: 5.1, y: 3.85, w: 4.45, h: 1.3 }, "還沒修好的一塊", "第 3 步 query 的點雲和文字，還是跟 gallery 同一個檔。所以含點雲的組合都還 95 以上，論文是 44。論文的 query 一定是另一份觀測，論文沒寫怎麼取。正在跑 P8：三個模態全換成不同觀測。", theme.accent, 10);
}

{
  const s = base("主因：query 讀的是 gallery 同一份資料，等於拿答案對答案", "審計項目 E1");
  headline(s, "舊程式把同一個 dict、同一批張量同時餵給兩座塔。模型只要學會「照抄」，就能全對。", 1.12, theme.accent);
  img(s, "eq_trivial.png", { x: 0.6, y: 1.95, w: 8.8 });
  card(s, { x: 0.45, y: 2.85, w: 2.95, h: 2.25 }, "白話", "考試時題目卷和答案卷是同一張紙。學生不用懂題目，把答案抄過來就 100 分。Eq. 5 在這種情況下有一個「偷懶解」：兩座塔輸出一模一樣。", theme.accent, 10.5);
  card(s, { x: 3.55, y: 2.85, w: 2.95, h: 2.25 }, "證據", "實測：query 向量和 gallery 向量的 cosine = 0.9989，三個都給 R@1 = 99.98。模型真的找到了偷懶解。", theme.secondary, 10.5);
  card(s, { x: 6.65, y: 2.85, w: 2.9, h: 2.25 }, "為什麼確定論文不是這樣", "論文三個都給只有 51.7。如果論文也是同一份資料，會跟我們一樣接近 100。所以論文的 query 一定是另一份觀測。論文沒寫怎麼取得。", theme.secondary, 10.5);
}

{
  const s = base("其他六個問題（也修了）");
  table(s, [["#", "之前的做法", "為什麼錯", "怎麼知道的", "修法"],
            ["2", "餵給文字塔的字串，把 160 字的自由描述放最前面", "論文 Figure 1、Figure 2 的文字輸入是「類別 + 結構化欄位」的短填表，沒有長描述", "換成填表，只給文字 58 掉到 8", "填表模板：{類別} made of {材質}, roughly {長}×{寬}×{高} cm, {擺放位置}"],
            ["3", "三個模態的向量直接丟進 Fusion", "訓練後點雲向量長度 139，文字 37、圖片 40。點雲在數值上壓過其他兩個", "量向量長度", "每個模態先做 L2 正規化再進 Fusion"],
            ["4", "分析工具把 gallery 的點雲餵成全零", "全零不等於「沒有」，結論全部無效", "抓到後全部撤回", "工具改成拒跑，加測試釘住"],
            ["5", "Stage 2 建模型時漏帶兩個 Stage 1 的設定", "Stage 2 的塔跟它的父模型不是同一個設定", "審計 F1", "修好並加測試"],
            ["6", "Stage 2 批次尾巴只剩 1 個樣本", "1 個樣本沒有負例，loss 變 0", "審計 F2", "少於 8 個樣本的批次丟掉"],
            ["7", "τ、評分方式、同分規則各處不一樣", "數字互相比不了", "審計", "全部走同一個評估器"]],
        { x: 0.45, y: 1.2, w: 9.1 }, { colW: [0.3, 2.3, 2.7, 1.5, 2.3], fontSize: 8.5, rowH: 0.52 });
}

// ================================================================ 04
section("04", "改了什麼、Stage 1 做了哪些測試", "改法對應論文哪一句；八個版本一次只換一個東西；目前數據；為什麼選 P1。");

{
  const s = base("改了什麼（對應論文哪一句）");
  table(s, [["改變", "為什麼（論文依據）", "怎麼做", "證據等級"],
            ["query 改讀「第二份觀測」", "Table 1 三個都給 51.7：同一份會 100", "圖片：12 張裡挑 1 張給 query，gallery 用 12 張平均；文字：另一段描述；點雲：重新取樣", "論文數字反推"],
            ["文字改填表", "Figure 2 的 13 個欄位、Figure 1 的文字輸入", "固定句型，只填欄位，不重新生成", "論文圖"],
            ["進 Fusion 前 L2 正規化", "論文沒寫；量到點雲壓過其他模態", "每個模態各自正規化", "我們的選擇"],
            ["單一評估器", "R@1 / R@5", "float64 cosine，同分算輸；另加 level、shape 兩個「離論文多遠」的量尺", "量測"],
            ["Stage 2 塔對齊父模型", "§2.6：凍結 gallery、只訓 query Fusion + ESSGNN", "建模型時鏡射 Stage 1 設定", "bug 修正"],
            ["Stage 2 配方", "§2.6 叫 fine-tuning，沒給學習率", "5e-5、warmup 10%、cosine；query 遮罩模式可選", "我們的選擇"]],
        { x: 0.45, y: 1.2, w: 9.1 }, { colW: [1.9, 2.7, 3.3, 1.2], fontSize: 9.5, rowH: 0.5 });
}

{
  const s = base("Stage 1 的測試設計：一次只換一個東西，看成績單的形狀會不會變");
  headline(s, "八個版本，同一份資料、同一組訓練設定、同一個評估器。差別只有名字後面那一件事。");
  table(s, [["版本", "跟前一版差在哪", "白話"],
            ["pilot10b", "舊做法（同一份資料、長描述、12 張平均、不正規化）", "對照組"],
            ["**P1", "填表文字 + query 只給 1 張圖 + 進 Fusion 前正規化", "三個修正一起上"],
            ["P3", "P1，但 12 張圖各給一個 token 進 Fusion（不平均）", "圖片給更細"],
            ["P4", "P1，但兩座塔共用一份 Fusion", "論文 Figure 1 畫的是一份"],
            ["P5", "文字用另一段描述 + 點雲重新取樣 + 1 張圖", "三個模態都換第二份"],
            ["P6", "P1，但每步隨機換一張圖", "圖片不固定"],
            ["P7", "P1，但關掉正規化", "驗證正規化的作用"]],
        { x: 0.45, y: 1.95, w: 9.1 }, { colW: [1.0, 5.6, 2.5], fontSize: 10, rowH: 0.34 });
  para(s, "共同設定：訓練 31,985 / 驗證 4,569；AdamW，學習率 5e-4，weight decay 0.1，warmup 1 代再 cosine 降；batch 64；10 代；τ 0.5；30% 遮罩；seed 20260816。每代在驗證集測七格 R@1 平均，取最高那代（都是第 9 代）。", { x: 0.45, y: 4.75, w: 9.1, h: 0.5 }, 9.5, GREY);
}

{
  const s = base("目前數據：八個版本對論文的 14 格（D 協定，R@1）");
  table(s, [["版本", "只給文字", "只給圖片", "只給點雲", "文字+圖片", "文字+點雲", "圖片+點雲", "三個都給", "level", "shape"],
            ["**論文", "**13.8", "**11.7", "**75.1", "**17.2", "**44.5", "**45.8", "**51.7", "0", "0"],
            ["**P1", "11.6", "29.7", "66.6", "67.5", "95.6", "77.8", "98.1", "0.59", "0.41"],
            ["P3", "10.4", "24.6", "61.1", "59.7", "94.6", "72.7", "98.0", "0.56", "0.40"],
            ["P4", "12.0", "25.0", "52.3", "58.5", "94.4", "65.8", "98.0", "0.54", "0.41"],
            ["P5", "14.3", "49.4", "88.5", "59.5", "92.8", "94.0", "95.5", "0.66", "0.40"],
            ["P6", "12.8", "28.8", "67.3", "67.1", "96.9", "77.3", "98.2", "0.58", "0.40"],
            ["P7", "9.5", "36.1", "69.1", "64.6", "95.6", "81.7", "98.5", "0.61", "0.44"],
            ["pilot10b", "58.0", "84.6", "78.8", "96.5", "99.6", "94.1", "100.0", "0.91", "0.57"]],
        { x: 0.45, y: 1.15, w: 9.1 }, { colW: [1.0, 0.95, 0.95, 0.95, 1.0, 1.0, 1.0, 1.0, 0.65, 0.6], fontSize: 10, rowH: 0.3 });
  card(s, { x: 0.45, y: 4.0, w: 4.45, h: 1.15 }, "level 和 shape 是什麼", "level：14 格整體比論文高多少（越小越好）。shape：把整體高低扣掉後，七格的形狀像不像論文（越小越像）。", theme.secondary, 10.5);
  card(s, { x: 5.1, y: 4.0, w: 4.45, h: 1.15 }, "怎麼讀", "只給文字、只給點雲已經接近論文。歪掉的全是「含點雲的組合」：論文加線索會掉，我們加線索一路升到 98。", theme.accent, 10.5);
}

{ const s = base("圖：八個版本 vs 論文（線越貼近黑線越像）"); img(s, "report_fig1_stage1_arms_D.png", { x: 0.9, y: 1.15, h: 4.2 }); }

{
  const s = base("結論：換架構，形狀都不動");
  headline(s, "七個新版本的 shape 全在 0.40～0.44。Fusion 一份或兩份、圖片平均或逐張、正規化開或關、視角固定或隨機：形狀都一樣。");
  img(s, "report_fig2_level_shape.png", { x: 0.45, y: 1.95, h: 3.1 });
  card(s, { x: 6.5, y: 1.95, w: 3.05, h: 3.15 }, "這代表什麼", "如果問題在架構，換架構形狀就會變。它沒變，所以問題不在架構。\n\n問題在資料：query 的觀測跟 gallery 太像。第 04 章證明這件事。", theme.accent, 11);
}

{
  const s = base("為什麼選 P1 當主線");
  const cards = [["1. 乾淨", "一次只動一個變數。三個修正各對應論文一句話：Figure 2 填表、Figure 1 一張圖、量到的正規化問題。", theme.secondary],
                 ["2. 單模態最接近論文", "只給文字 11.6（論文 13.8）；只給點雲 66.6（論文 75.1）。", theme.secondary],
                 ["3. 其他版本沒有更好", "P4 只給點雲掉到 52；P3、P6、P7 跟 P1 同一族；P5 各格更高（重新取樣的點雲跟原檔相似度 0.99，還是太像）。", theme.accent],
                 ["4. 可以被推翻", "P1 是 Stage 2 的父模型。你已授權：學習率、幾代、query 觀測都在重跑，結果會確認或推翻 P1。", theme.light]];
  cards.forEach((c, i) => card(s, { x: 0.45 + (i % 2) * 4.65, y: 1.2 + Math.floor(i / 2) * 1.95, w: 4.45, h: 1.8 }, c[0], c[1], c[2], 11));
}

// ================================================================ 04
section("05", "融合分數為什麼還太高", "先排除 ULIP-2 被改；再用數字證明是 query 跟 gallery 太像；最後用實驗確認。");

{
  const s = base("先排除：ULIP-2 沒有被動");
  headline(s, "用官方釋出的權重、照官方的評估程式，在我們的資料上做零樣本分類，數字跟 ULIP-2 論文一樣。");
  img(s, "report_fig6_ulip2_zero_shot.png", { x: 0.45, y: 1.95, h: 3.1 });
  bullets(s, ["**vendored 的 ULIP 程式碼 17 個檔案跟上游逐 byte 相同",
              "相容補丁只做三件事：舊 torch 的 import 轉接；CUDA 擴充編不起來，最遠點取樣改純 torch 同一個演算法；knn 放替身",
              "**零樣本 Objaverse-LVIS 分類：top-1 50.9、top-5 79.3；論文 Table 10 是 50.6、79.1",
              "結論：backbone、點雲前處理、文字塔就是論文的 ULIP-2。分數高不是這裡來的"], { x: 5.6, y: 1.95, w: 4.0, h: 3.2 }, 11);
}

{
  const s = base("ULIP-2 的「拉近三個模態」是什麼意思");
  headline(s, "拉近 = 自己的文字、圖片比別人的更靠近自己的點雲。不是三個點疊成一個點。");
  table(s, [["實測（釋出 ULIP-2，1,024 個物件）", "跟自己的點雲", "跟別人的點雲（平均）"],
            ["文字向量", "0.29", "0.07"], ["圖片向量", "0.46", "0.11"], ["點雲向量", "1.00（同一個檔）", "最像的別人 0.59"]],
        { x: 0.45, y: 1.95, w: 5.2 }, { colW: [2.4, 1.4, 1.4], fontSize: 10.5, rowH: 0.34 });
  card(s, { x: 5.85, y: 1.95, w: 3.7, h: 1.45 }, "所以「只給文字」為什麼低", "文字對自己 0.29、對別人平均 0.07，差距很小；總有某個很像的別人比自己更近。文字是「分不清」，不是「指向別人」。", theme.secondary, 10.5);
  card(s, { x: 0.45, y: 3.5, w: 9.1, h: 1.6 }, "ULIP-2 訓練時做了什麼（論文 §3）", "CLIP 的文字塔和影像塔鎖住不動，只訓練點雲編碼器：讓點雲向量靠近自己的文字和圖片、遠離別人的。所以三個模態在同一個空間裡，但彼此還是分開的點。這是 MetaFind 拿來用的現成模型。", theme.accent, 11);
}

{
  const s = base("關鍵：query 的點雲跟 gallery 的點雲是同一個檔");
  headline(s, "加文字、加圖片確實把 query 拉離自己，但也一樣拉離別人。差距一直在，名次翻不了，所以只會越加越高。", 1.12, theme.accent);
  img(s, "fig_own_vs_other.png", { x: 0.45, y: 1.95, h: 3.1 });
  img(s, "eq_mean.png", { x: 0.45, y: 5.1, w: 9.1 });
  card(s, { x: 6.5, y: 1.95, w: 3.05, h: 3.1 }, "白話", "拿身分證照片去比對身分證照片，一定 100 分。旁邊多放一句描述、多放一張生活照，只是把所有人的分數一起降一點，第一名還是同一個。\n\n不用 Fusion、不訓練、單純把向量平均，也一樣越加越高（文字+點雲 98.6）。", theme.accent, 10.5);
}

{
  const s = base("那 P1 的「只給點雲」為什麼不是 100？");
  headline(s, "因為 gallery 存的是三模態融合過的向量；query 只給點雲時是「點雲 + 兩個 mask token」進 Fusion，走的路不一樣。");
  img(s, "fig_shape_paper_vs_p1.png", { x: 0.45, y: 1.95, h: 3.1 });
  card(s, { x: 6.5, y: 1.95, w: 3.05, h: 1.5 }, "只給點雲 66.6", "query 少了兩個模態，Fusion 的輸出跟 gallery 那個三模態的向量對不齊，掉分。", theme.secondary, 10.5);
  card(s, { x: 6.5, y: 3.55, w: 3.05, h: 1.5 }, "三個都給 98.1", "query 跟 gallery 走同一條路，而且點雲、文字是同一個檔，只有圖片差一張。幾乎自己對自己。", theme.accent, 10.5);
}

{
  const s = base("實驗確認：只換 query 的點雲，其他不動（P1 直接重評，不重訓）");
  headline(s, "把 query 的點雲換成同一個物件的「不同看法」：去顏色、加雜訊、只留一半。點雲分數掉很快，但「三個都給」永遠高於「只給點雲」。");
  img(s, "report_fig4_query_pc_observation.png", { x: 0.3, y: 1.95, h: 3.0 });
  table(s, [["query 點雲", "像原檔幾成", "只給點雲", "文字+點雲", "三個都給"],
            ["原檔", "1.00", "66.6", "95.6", "98.1"], ["重新取樣", "0.997", "63.8", "94.6", "97.3"],
            ["去顏色", "0.83", "8.9", "20.4", "31.4"], ["加雜訊", "0.82", "9.1", "38.8", "58.7"],
            ["只留一半", "0.81", "5.4", "28.0", "46.6"], ["**論文", "?", "**75.1", "**44.5", "**51.7"]],
        { x: 6.4, y: 1.95, w: 3.2 }, { colW: [0.9, 0.7, 0.55, 0.55, 0.5], fontSize: 8.5, rowH: 0.27 });
  para(s, "兩個發現：(1) P1 的點雲塔很脆，稍微動一下就從 66 掉到個位數，釋出的 ULIP-2 同樣雜訊還有 45。(2) 光弄壞點雲做不出論文的形狀，文字和圖片也要是「不同觀測」。這就是接下來的 P8 實驗。", { x: 0.45, y: 4.95, w: 9.1, h: 0.55 }, 10, GREY);
}

{
  const s = base("訓練夠不夠？（10 代會不會太少）", "論文沒寫幾代。250 是 ULIP 官方程式碼的預設值，不是 MetaFind 說的");
  headline(s, "不是訓練長短的問題：P1 從第 0 代起「三個都給」就高於「只給點雲」，越練七格一起往上。");
  img(s, "report_fig3_P1_epochs.png", { x: 0.3, y: 1.95, h: 3.1 });
  bullets(s, ["舊版本跑 44 代：七格全 90 以上",
              "ULIP 官方：固定 250 代，cosine 降到底；每代在 ModelNet40 驗證，取最好那代；沒有 early stopping。我們選法一樣，只差代數",
              "**正在驗證：P1 跑 25 代（進行中）；學習率掃 1e-4 / 1e-3 / 3e-3（排隊）"], { x: 6.2, y: 1.95, w: 3.4, h: 3.2 }, 10.5);
}

// ================================================================ 05
section("06", "Stage 2 做了什麼", "房間層級微調：資料、配方、四次跑的結果、候選版本。");

{
  const s = base("Stage 2 的設定");
  table(s, [["項目", "值", "依據"],
            ["場景", "ProcTHOR-10k 訓練集前 1,500 間房（全部 9,600 間還沒跑）", "先小規模驗證流程"],
            ["樣本", "99,945 個：房間裡每個物件當一次 query，任務是找出它是哪個資產", ""],
            ["資產庫", "1,439 個 ProcTHOR 資產，用 P1 編碼", ""],
            ["凍結", "gallery 塔、ULIP-2 全部凍結；只訓 query 的 Fusion 和 ESSGNN", "論文 §2.6"],
            ["λ 初值", "0.1 × Fusion 向量長度的中位數 = 93.46", "我們的選擇"],
            ["場景 dropout", "每個 batch 30% 不給 layout", "論文"],
            ["損失", "Eq. 7、8 雙向，取平均，τ 0.5", "論文"],
            ["學習率", "先導：跟 Stage 1 一樣 5e-4 平坦；S2-C / S2-D：5e-5，warmup 10%，cosine 降，1 代", "論文只說 fine-tuning"],
            ["Table 1 的 w/ ESSGNN 列", "把 Stage 2 訓好的 query 頭疊回 P1，在 Objaverse 上評，layout 關掉", "論文 §3.2"]],
        { x: 0.45, y: 1.2, w: 9.1 }, { colW: [1.6, 5.7, 1.8], fontSize: 9.5, rowH: 0.36 });
}

{
  const s = base("Stage 2 結果");
  headline(s, "layout 分支在「精確找出同一個資產」上幫不上忙（跟論文 w/ 比 w/o 低的方向一致）。掉多少由學習率配方決定。");
  img(s, "report_fig5_stage2.png", { x: 0.3, y: 1.95, h: 2.3 });
  table(s, [["版本", "query 給什麼", "學習率", "ProcTHOR 找資產 R@1（Stage 1 頭 / 關 layout / 開 layout）", "w/ ESSGNN 列（C 協定，七格 R@1）"],
            ["先導 2", "三個都給", "5e-4 平坦", "82.4 / 24.2 / 23.5", "10.1 / 15.2 / 49.2 / 22.2 / 56.4 / 50.5 / 58.9"],
            ["**S2-C", "只給文字（論文 Figure 1 的 query 長相）", "5e-5 warmup cosine", "10.3 / 12.4 / 12.2", "22.5 / 37.7 / 74.3 / 62.8 / 86.1 / 80.1 / 88.2"],
            ["S2-D", "三個都給", "5e-5 warmup cosine", "82.4 / 36.8 / 32.8", "24.9 / 36.1 / 71.2 / 58.2 / 80.9 / 73.0 / 80.1"]],
        { x: 0.45, y: 4.3, w: 9.1 }, { colW: [0.7, 2.1, 1.4, 2.3, 2.6], fontSize: 8, rowH: 0.24 });
  para(s, "P1 父模型在 C 協定：34.7 / 56.9 / 86.1 / 87.6 / 99.0 / 92.7 / 99.7。論文 w/ 除以 w/o 是 0.82～0.93；S2-C 除以父是 0.65～0.88。λ 三次都幾乎不動（93.46 到 93.43）。S2-C 是候選，不是定案。", { x: 0.45, y: 5.28, w: 8.7, h: 0.3 }, 8.5, GREY);
}

// ================================================================ 06
section("07", "接下來計畫", "五段排程、每段的判斷規則、要你決定的事。");

{
  const s = base("目前情況總結");
  const cards = [["架構與公式", "Stage 1 35 項、Stage 2 Eq. 2 到 8 逐項對過論文。唯一的設計偏離：CLIP 文字／影像塔凍結（論文 §3.4 說全部微調較好；這張卡記憶體不夠）。", theme.secondary],
                 ["ULIP-2", "沒被動。零樣本分類重現論文數字。", theme.secondary],
                 ["Stage 1", "單模態分數已接近論文。融合分數偏高的原因確定：query 觀測跟 gallery 太像。不是架構、不是訓練長短、不是 ULIP-2。", theme.accent],
                 ["主線與授權", "主線 P1；Stage 2 候選 S2-C；兩者都可被接下來的實驗推翻。你已授權：設定（含學習率）由論文證據和實驗證據決定；只有資料集允許先不一致。", theme.light]];
  cards.forEach((c, i) => card(s, { x: 0.45 + (i % 2) * 4.65, y: 1.2 + Math.floor(i / 2) * 1.95, w: 4.45, h: 1.8 }, c[0], c[1], c[2], 11));
}

{
  const s = base("排程（一張卡，一個接一個跑）");
  table(s, [["段", "做什麼", "狀態 / 預計", "看什麼決定下一步"],
            ["1", "P1 設定不變，10 代改 25 代", "跑中，約 08:00", "三個都給仍高於只給點雲 → 訓練長短不是原因，250 代不跑"],
            ["2", "學習率掃 1e-4 / 1e-3 / 3e-3，各 10 代（5e-4 已有）", "排隊，約 10:40", "驗證集七格平均選；另外報離論文多遠"],
            ["3", "P8：query 三個模態全換第二份觀測（另一段描述、一張圖、只留一半的點雲），用第 2 段選出的學習率", "排隊，約 11:45", "三個都給開始低於只給點雲 → 方向對，再試去顏色、加雜訊"],
            ["4", "CLIP 文字／影像塔開最後幾層（論文 full encoder fine-tuning）", "之後", "先量記憶體；全開這張卡吃不下"],
            ["5", "定案 Stage 1；Stage 2 三個版本在新父模型上重跑；全部 9,600 間房", "之後", ""]],
        { x: 0.45, y: 1.2, w: 9.1 }, { colW: [0.4, 4.3, 1.4, 3.0], fontSize: 10, rowH: 0.5 });
  card(s, { x: 0.45, y: 4.3, w: 9.1, h: 0.85 }, "要你決定的", "ProcTHOR 切分：80/20（論文寫法）還是官方 10k/1k/1k？要不要用 GPT-4o 當場景評審？最後評估要不要解封測試集？", theme.accent, 10.5);
}

{
  const s = pres.addSlide(); s.background = { color: theme.primary };
  s.addText("一句話總結", { x: 0.6, y: 0.5, w: 8, h: 0.7, fontSize: 28, fontFace: ZH, color: theme.light, bold: true, margin: 0 });
  const pts = ["模型、公式、ULIP-2 都對過了，跟論文一樣。", "第一次分數太高是因為拿答案對答案；改掉後單模態已接近論文。", "融合分數還太高，原因是 query 的點雲和文字跟 gallery 是同一個檔。這是資料層的事，正在用「不同觀測」重跑。", "P1 是目前主線、S2-C 是 Stage 2 候選；學習率、代數、query 觀測的實驗會確認或推翻它們。"];
  s.addText(pts.map((t, i) => ({ text: (i + 1) + ".  " + t, options: { fontSize: 16, fontFace: ZH, color: "FFFFFF", paraSpaceAfter: 14, breakLine: true } })), { x: 0.6, y: 1.5, w: 8.8, h: 3.4, valign: "top", margin: 0 });
  badge(s);
}

pres.writeFile({ fileName: "./output/MetaFind_report_20260904.pptx" }).then((f) => console.log("wrote", f));

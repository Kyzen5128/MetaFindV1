# Report deck (2026-09-04)

Built with the pptx-generator skill (PptxGenJS). To rebuild:

```
mkdir -p output/look/pptx/slides/imgs && cd output/look/pptx && npm install pptxgenjs
cp output/look/report_fig*.png output/look/ulip2_pull_explainer.png output/look/pptx/slides/imgs/
cp tools/report_pptx/compile.js output/look/pptx/slides/ && cd output/look/pptx/slides && node compile.js
```

Charts: tools/probes/draw_report_charts.py, tools/probes/draw_ulip2_pull_explainer.py. Output: output/look/MetaFind_report_20260904.pptx

"""Build an editable research deck and speaker handout from the reviewed content JSON.

Run with the existing MetaFind Python environment. No model/data producers run here.
"""
from pathlib import Path
import hashlib,json,math,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR,PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement

ROOT=Path(__file__).resolve().parents[1]
CONTENT=Path(__file__).with_name('ppt_content.json')
ASSETS=ROOT/'assets';ASSETS.mkdir(exist_ok=True)
FONT='Noto Sans CJK TC'
INK='183247';TEAL='137D88';ORANGE='CE663F';MUTED='657A88';BG='F5F7F8';PALE='E6F1F0';LINE='D9E2E7';WHITE='FFFFFF'
COLORS=[INK,TEAL,ORANGE,'8E9FAA','8778A9']
prs=Presentation();prs.slide_width=Inches(13.333333);prs.slide_height=Inches(7.5)
prs.core_properties.title='MetaFind｜Table 1 評估與 Stage 2'
prs.core_properties.subject='Notion 研究指南簡報；附逐頁講稿'
prs.core_properties.author='MetaFindV1'
prs.core_properties.keywords='MetaFind, Table 1, Stage 2, reproduction, 自訂評估'
shape_boxes=[]

def clean(v):
    return re.sub(r'`([^`]+)`',r'\1',str(v)).replace('**','')

def setfont(run,size,color=INK,bold=False):
    run.font.name=FONT;run.font.size=Pt(size);run.font.bold=bold;run.font.color.rgb=RGBColor.from_string(color)
    rp=run._r.get_or_add_rPr()
    for tag in ('a:ea','a:cs'):
        node=rp.find(tag, rp.nsmap)
        if node is None:node=OxmlElement(tag);rp.append(node)
        node.set('typeface',FONT)

def text(slide,value,x,y,w,h,size=22,color=INK,bold=False,align=PP_ALIGN.LEFT):
    shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=shape.text_frame;tf.word_wrap=True;tf.margin_left=Inches(.015);tf.margin_right=Inches(.015)
    tf.margin_top=Inches(.01);tf.margin_bottom=Inches(.01)
    for i,line in enumerate(clean(value).split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.alignment=align
        p.space_before=Pt(0);p.space_after=Pt(2);p.line_spacing=1.12
        r=p.add_run();r.text=line;setfont(r,size,color,bold)
    shape_boxes.append({'slide':len(prs.slides),'text':clean(value),'box':[x,y,w,h],'size':size})
    return shape

def box(slide,x,y,w,h,fill=WHITE,line=None,rounded=True):
    sh=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(fill)
    if line:sh.line.color.rgb=RGBColor.from_string(line);sh.line.width=Pt(.7)
    else:sh.line.fill.background()
    if rounded:sh.adjustments[0]=.08
    return sh

def line(slide,x1,y1,x2,y2,color=LINE,width=1.3):
    sh=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2))
    sh.line.color.rgb=RGBColor.from_string(color);sh.line.width=Pt(width)
    return sh

def short_source(s):
    v=s.get('sources',[])
    if not v:return '來源：指定 Notion 頁面；完整來源見備忘稿'
    first=clean(v[0]);first=re.sub(r'/home/kyzen/MetaFindV1/','',first)
    return '來源：'+first[:116]

def base(s,n,total):
    sl=prs.slides.add_slide(prs.slide_layouts[6]);sl.background.fill.solid();sl.background.fill.fore_color.rgb=RGBColor.from_string(BG)
    box(sl,0,0,.15,7.5,TEAL,rounded=False)
    text(sl,str(s.get('section','RESEARCH BRIEF')).upper(),.57,.24,11.4,.27,11,TEAL,True)
    title=clean(s['title']);size=31 if len(title)<=30 else 27
    text(sl,title,.57,.69,12.16,1.01,size,INK,True)
    line(sl,.6,1.62,12.73,1.62)
    box(sl,.6,6.47,12.1,.42,PALE,rounded=False)
    take=clean(s['takeaway']);text(sl,take,.75,6.50,11.8,.34,15 if len(take)>48 else 17,TEAL,True)
    footer=text(sl,short_source(s),.60,7.08,11.1,.20,8.2,MUTED)
    footer.click_action.hyperlink.address=json.loads(Path(__file__).with_name('notion_provenance.json').read_text())['url']
    text(sl,f'{n:02d} / {total:02d}',11.78,7.05,.93,.26,11,MUTED,align=PP_ALIGN.RIGHT)
    return sl

def bullets(sl,items,x,y,w,step=.95,size=22):
    for i,item in enumerate(items):
        text(sl,'•',x,y+i*step,.2,.4,size,TEAL,True)
        text(sl,item,x+.3,y+i*step,w-.3,step-.06,size,INK)

def cards(sl,s):
    items=s.get('bullets',[])
    cols=2 if len(items)>2 else max(len(items),1)
    rows=math.ceil(max(len(items),1)/cols)
    w=(12.1-.28*(cols-1))/cols;h=(4.48-.25*(rows-1))/rows
    for i,item in enumerate(items):
        x=.6+(i%cols)*(w+.28);y=1.82+(i//cols)*(h+.25)
        box(sl,x,y,w,h,WHITE,LINE)
        text(sl,f'{i+1:02d}',x+.23,y+.18,.73,.42,23,TEAL,True)
        text(sl,item,x+.24,y+.82,w-.48,h-.97,23 if len(clean(item))<45 else 20,INK)

def draw_table(sl,s):
    data=s['table'];headers=data['headers'];rows=data['rows'];count=len(headers)
    ratios=data.get('widths')
    if not ratios or len(ratios)!=count:
        ratios=([2.5]+[1]*(count-1)) if count>=6 else ([1.25]+[1]*(count-1))
    widths=[12.1*float(v)/sum(ratios) for v in ratios]
    top=s.get('_table_top',1.85);available=s.get('_table_height',3.67 if s.get('bullets') else 4.4)
    heights=[.54]+[(available-.54)/max(len(rows),1)]*len(rows)
    y=top
    for ri,row in enumerate([headers]+rows):
        x=.6
        assert len(row)==count,(s['id'],ri,row)
        for ci,value in enumerate(row):
            fill=INK if ri==0 else (WHITE if ri%2 else 'EBF0F3')
            box(sl,x,y,widths[ci],heights[ri],fill,rounded=False)
            raw=clean(value);size=s.get('_table_size',14 if count>=6 else 17)
            if len(raw)>42:size=14
            if len(raw)>72:size=12.5
            shape=text(sl,raw,x+.10,y+.06,widths[ci]-.2,heights[ri]-.1,size,WHITE if ri==0 else INK,ri==0 or ci==0)
            shape.text_frame.vertical_anchor=MSO_ANCHOR.MIDDLE
            x+=widths[ci]
        y+=heights[ri]
    if s.get('bullets') and not s.get('_no_table_bullets'):bullets(sl,s['bullets'][:2],.7,5.70,11.9,.31,14)

def draw_chart(sl,s):
    data=s['chart'];cats=data['categories'];series=data['series'];n=len(cats);m=len(series)
    prop=font_manager.FontProperties(family=FONT)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[FONT],'axes.unicode_minus':False})
    fig,ax=plt.subplots(figsize=(9.2,5.0),dpi=180);fig.patch.set_facecolor('#'+BG);ax.set_facecolor('#'+BG)
    group=.78;bar=group/m
    for j,ser in enumerate(series):
        ys=[i-group/2+(j+.5)*bar for i in range(n)]
        ax.barh(ys,ser['values'],height=bar*.86,label=clean(ser['name']),color='#'+COLORS[j%len(COLORS)])
        for yy,v in zip(ys,ser['values']):
            ax.text(v+1 if v<94 else v-1,yy,f'{v:.2f}',va='center',ha='left' if v<94 else 'right',fontsize=15.5,color='#'+INK if v<94 else '#'+WHITE)
    ax.set_yticks(range(n),[clean(v) for v in cats],fontproperties=prop,fontsize=16)
    ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel(data.get('unit','R@1 (%)'),fontsize=12)
    ax.tick_params(axis='both',length=0,labelcolor='#'+MUTED);ax.grid(axis='x',color='#'+LINE,lw=.6);ax.set_axisbelow(True)
    for sp in ax.spines.values():sp.set_visible(False)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=min(m,2),frameon=False,prop={'family':FONT,'size':12.5})
    fig.tight_layout(pad=1.4)
    filename=f"slide_{int(s['id']):02d}_chart"
    png=ASSETS/(filename+'.png');fig.savefig(png,facecolor=fig.get_facecolor(),bbox_inches='tight')
    fig.savefig(ASSETS/(filename+'.pdf'),facecolor=fig.get_facecolor(),bbox_inches='tight');plt.close(fig)
    iw,ih=Image.open(png).size;scale=min(8.1/iw,4.44/ih);pw,ph=iw*scale,ih*scale
    sl.shapes.add_picture(str(png),Inches(.6+(8.1-pw)/2),Inches(1.78+(4.44-ph)/2),width=Inches(pw),height=Inches(ph))
    box(sl,9.00,1.88,3.68,4.26,WHITE,LINE)
    bullets(sl,s.get('bullets',[])[:4],9.20,2.08,3.26,.81,16)
    note=data.get('note')
    if note:text(sl,note,9.23,5.46,3.13,.63,10.5,MUTED)

def draw_flow(sl,s):
    data=s['flow'];steps=data['steps'];count=len(steps)
    if count<=4:cols=count
    else:cols=3 if count<=6 else 4
    rows=math.ceil(count/cols);gap=.30;w=(12.1-gap*(cols-1))/cols;h=(s.get('_flow_height',4.18)-.40*(rows-1))/rows
    for i,item in enumerate(steps):
        if isinstance(item,str):item={'title':item,'detail':''}
        col=i%cols;row=i//cols;x=.6+col*(w+gap);y=s.get('_flow_top',1.96)+row*(h+.40)
        box(sl,x,y,w,h,WHITE,LINE)
        if s.get('_flow_compact'):
            text(sl,f'{i+1:02d}',x+.15,y+.12,.51,.35,16,TEAL,True)
            text(sl,item['title'],x+.78,y+.12,w-.95,.58,16,INK,True)
            if item.get('detail'):text(sl,item['detail'],x+.15,y+.82,w-.30,.51,13.5,MUTED)
        else:
            text(sl,f'{i+1:02d}',x+.15,y+.12,.56,.36,18,TEAL,True)
            text(sl,item['title'],x+.15,y+.58,w-.30,.66,18 if w>=3 else 16,INK,True)
            if item.get('detail'):text(sl,item['detail'],x+.15,y+1.23,w-.30,max(.44,h-1.31),14,MUTED)
        if col<cols-1 and i<count-1:text(sl,'→',x+w+.02,y+.7,gap-.04,.42,20,TEAL,True,PP_ALIGN.CENTER)
    if data.get('note'):text(sl,data['note'],.7,6.19,11.8,.24,11,MUTED)

def cover(sl,s):
    sl.background.fill.solid()
    sl.background.fill.fore_color.rgb=RGBColor.from_string(INK)
    # Cover has its own composition; no scientific scores are invented here.
    text(sl,'METAFIND  /  REPRODUCTION BRIEF',.73,.59,11.8,.40,13,'67C5C4',True)
    text(sl,'Table 1 評估\n與 Stage 2',.72,1.55,7.65,1.94,43,WHITE,True)
    text(sl,'方法、證據與下一步',.75,3.79,7.6,.7,25,'C6D8E1')
    text(sl,'同一 UID 的檢索比較\n場景條件下的 layout 驗證',.76,5.01,7.1,.94,20,'C6D8E1')
    text(sl,'2026.09.08  ·  研究進度報告  ·  含逐頁備忘稿',.76,6.70,10.6,.35,12,'A9C2CD')
    for i,letter in enumerate(['T','I','PC']):
        box(sl,9.2,1.45+i*1.26,2.47,.85,'23495C',rounded=True)
        text(sl,letter,9.44,1.56+i*1.26,1.98,.5,26,'8ED3CF',True,PP_ALIGN.CENTER)
    line(sl,10.43,4.92,10.43,5.31,'67C5C4',2)
    box(sl,8.72,5.32,3.45,.87,TEAL)
    text(sl,'QUERY → GALLERY',8.96,5.56,2.98,.37,16,WHITE,True,PP_ALIGN.CENTER)

def code_slide(sl,s):
    code=s.get('code') or s.get('code_blocks') or s.get('commands')
    if isinstance(code,list):code='\n\n'.join(v.get('code',str(v)) if isinstance(v,dict) else str(v) for v in code)
    if isinstance(code,dict):code=code.get('text',code.get('code',str(code)))
    if not code:
        cards(sl,s);return
    box(sl,.6,1.83,12.1,4.4,INK)
    lines=str(code).splitlines();size=15 if len(lines)<=14 else 11.3
    tb=text(sl,code,.86,2.01,11.6,4.0,size,'E3F0F4')
    for p in tb.text_frame.paragraphs:
        p.line_spacing=1.10
        for r in p.runs:r.font.name='DejaVu Sans Mono'


def formula(sl,s):
    # An explicit example, never a data-dependent target-diagonal assumption.
    example=s.get('formula') or s.get('example')
    if not example:
        cards(sl,s);return
    if isinstance(example,dict):example='\n'.join(f'{k}: {v}' for k,v in example.items())
    box(sl,.6,1.86,12.1,2.15,INK)
    text(sl,example,.94,2.18,11.35,1.48,25,WHITE,True)
    bullets(sl,s.get('bullets',[])[:3],.83,4.43,11.75,.59,20)



def focused_layout(sl,s):
    sid=int(s['id'])
    if sid==3:
        text(sl,'ULIP2＋本地 MEAN  /  STAGE 1  /  STAGE 2 — LAYOUT OFF',.71,1.85,11.7,.40,18,TEAL,True)
        draw_table(sl,{**s,'_table_top':2.39,'_table_height':3.11,'_table_size':16,'_no_table_bullets':True})
        text(sl,'T   I   PC   T＋I   T＋PC   I＋PC   FULL',.75,5.76,11.7,.40,21,INK,True)
        text(sl,'每個 gallery 都是完整 T＋I＋PC；不同文字須經有效 token 檢查。',.76,6.15,11.7,.24,11.5,MUTED)
    elif sid==4:
        text(sl,'教學示例（非模型結果）',.73,1.86,11.7,.39,18,TEAL,True)
        draw_table(sl,{**s,'_table_top':2.36,'_table_height':1.66,'_table_size':17,'_no_table_bullets':True})
        text(sl,'rank = 1 + 更高分的非GT數 + 同分的非GT數',.77,4.34,11.7,.51,24,INK,True)
        text(sl,s['formula'],.77,5.05,11.7,.46,21,TEAL,True)
        text(sl,'以 float64 cosine 排名；小 gallery 的 R@5 可能缺乏鑑別力。',.77,5.77,11.7,.36,16,MUTED)
    elif sid==7:
        text(sl,'舊穩定 corpus 的部分 preflight：M = A + Q + E',.73,1.85,11.6,.36,18,TEAL,True)
        values=[('A  admitted','45,692'),('Q  真實失敗','339'),('E  人工排除','21')]
        for j,(label,value) in enumerate(values):
            x=.62+j*4.12;box(sl,x,2.34,3.86,1.08,WHITE,LINE)
            text(sl,label,x+.16,2.45,3.54,.29,13,MUTED)
            text(sl,value,x+.16,2.82,3.54,.47,28,INK,True)
        text(sl,'原始 M＝46,052；2% 只計 Q/M。Paper 在製資料仍 BLOCKED。',.75,3.67,11.7,.43,17,INK,True)
        draw_table(sl,{**s,'_table_top':4.40,'_table_height':1.40,'_table_size':14,'_no_table_bullets':True})
        text(sl,'本條 S1 使用整份 holdout 選模，已包含 test；G3 尚未接 live chain。',.77,6.00,11.7,.28,13,ORANGE,True)
    elif sid==8:
        box(sl,.6,1.83,12.1,.59,INK)
        text(sl,'LR 5e-5   ·   1 epoch   ·   batch 64   ·   τ 0.5   ·   scene dropout 0.3',.81,1.94,11.7,.34,18,WHITE,True)
        draw_flow(sl,{**s,'_flow_top':2.66,'_flow_height':3.25,'_flow_compact':True})
    elif sid==9:
        draw_table(sl,{**s,'_table_top':1.87,'_table_height':2.67,'_table_size':16,'_no_table_bullets':True})
        box(sl,.6,4.91,5.88,1.10,INK);box(sl,6.77,4.91,5.93,1.10,TEAL)
        text(sl,'1,805',.84,5.03,2.32,.49,31,WHITE,True)
        text(sl,'CPU tests passed',3.03,5.22,3.23,.38,15,WHITE)
        text(sl,'2,342',7.01,5.03,2.32,.49,31,WHITE,True)
        text(sl,'graph checks',9.32,5.22,3.05,.38,15,WHITE)
        text(sl,'Scene_scores 尚未執行 judge；Human 保持 INSUFFICIENT_EVIDENCE。',.75,6.16,11.7,.23,11.5,ORANGE,True)
    else:
        return False
    return True

def make():
    obj=json.loads(CONTENT.read_text());slides=obj['slides'];total=len(slides)
    manuscript=['# MetaFind：Table 1 評估與 Stage 2 — 逐頁講稿','',
        '10 頁精簡版；此稿也寫入 PPT 每頁的「備忘稿」。完整命令附在講稿後方，不增加投影片頁數。',
        '來源：'+json.loads(Path(__file__).with_name('notion_provenance.json').read_text())['url'],'']
    for n,s in enumerate(slides,1):
        hint=s.get('layout_hint','cards')
        if n==1 or hint=='cover':
            sl=prs.slides.add_slide(prs.slide_layouts[6]);cover(sl,s)
        else:
            sl=base(s,n,total)
            if focused_layout(sl,s):pass
            elif s.get('chart'):draw_chart(sl,s)
            elif s.get('table'):draw_table(sl,s)
            elif s.get('flow'):draw_flow(sl,s)
            elif hint=='code' or s.get('code') or s.get('commands'):code_slide(sl,s)
            elif hint=='formula':formula(sl,s)
            else:cards(sl,s)
        sources='\n'.join(str(v) for v in s.get('sources',[]))
        notes=f"第 {n:02d} 頁｜{s['title']}\n建議時間：{s.get('duration_seconds',70)} 秒\n\n{s['notes']}\n\n本頁重點：{s['takeaway']}\n\n來源與界線\n{sources}"
        sl.notes_slide.notes_text_frame.text=notes
        for p in sl.notes_slide.notes_text_frame.paragraphs:
            for r in p.runs:setfont(r,12,INK)
        manuscript += [f"## {n:02d}｜{s['title']}",'',f"**建議時間：{s.get('duration_seconds',70)} 秒。**",'',s['notes'],'',f"**本頁重點：** {s['takeaway']}",'','來源：','']+['- '+str(v) for v in s.get('sources',[])]+['']
    out=ROOT/'MetaFind_Table1_Stage2_含講稿.pptx';prs.save(out)
    manuscript += ['# 操作附錄（不計入 10 頁投影片）','','以下保留 Notion 的命令模板。請先準備真實資料、UID 清單、外部描述與 checkpoint；文件中的 /path/to 是待指定路徑。本次簡報製作未執行這些訓練／評估命令。','']
    source=Path(__file__).with_name('notion_source.md').read_text()
    parts=re.split(r'(```bash\n[\s\S]*?\n```)',source)
    for i,part in enumerate(parts):
        if not part.startswith('```bash'):continue
        headings=re.findall(r'^#{2,3} (.+)$',parts[i-1],re.M)
        manuscript += ['## '+(headings[-1] if headings else '操作命令'),'',part,'']
    (ROOT/'逐頁講稿.md').write_text('\n'.join(manuscript).rstrip()+'\n')
    (ROOT/'source/layout_boxes.json').write_text(json.dumps(shape_boxes,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'source/build_record.json').write_text(json.dumps({'slides':total,'notes_slides':sum(bool(sl.notes_slide.notes_text_frame.text.strip()) for sl in prs.slides),'content_sha256':hashlib.sha256(CONTENT.read_bytes()).hexdigest(),'pptx_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'font':FONT,'size':'16:9','environment':'existing MetaFind python; python-pptx 1.0.2; matplotlib; no model producers'},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pptx':str(out),'slides':total,'bytes':out.stat().st_size},ensure_ascii=False))

if __name__=='__main__':make()

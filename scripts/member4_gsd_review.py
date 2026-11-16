"""Render GSD review materials and validate independently supplied human decisions."""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.member4_gsd_prepare import ROOT, DATA, MANIFESTS, read
from transparent_seg.data import decode_mask, read_manifest
from transparent_seg.prepare import sha256_file, write_json


def font(size=18):
    return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size)


def load_views(row, base, encoding):
    with Image.open(Path(base) / row['image']) as im:
        image = im.convert('RGB')
    y, valid = decode_mask(Path(base) / row['mask'], encoding, [])
    mask = Image.fromarray((y * 255).astype(np.uint8)).convert('RGB')
    rgb = np.asarray(image).copy().astype(np.float32)
    foreground = (y > .5) & valid
    rgb[foreground] = .55 * rgb[foreground] + .45 * np.array([0, 220, 160])
    return image, mask, Image.fromarray(rgb.astype(np.uint8))


def paste_fit(canvas, image, box):
    x, y, w, h = box
    im = image.copy(); im.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas.paste(im, (x + (w-im.width)//2, y + (h-im.height)//2))


def label_panel(row):
    canvas = Image.new('RGB', (1200, 500), 'white'); draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"row={row['source_row']}  {row['source_id']}", font=font(), fill='black')
    draw.text((12, 36), f"原尺寸 {row['width']}×{row['height']} | 0=背景，255=玻璃 | 人工复核：待填写", font=font(), fill='black')
    for col, (im, title) in enumerate(zip(load_views(row, DATA, 'binary_0255'), ('原图', '原标签（白色前景）', '标签叠图'))):
        draw.text((col*400+12, 68), title, font=font(), fill='black')
        paste_fit(canvas, im, (col*400+8, 100, 384, 390))
    return canvas


def cross_panel(pair, a, b):
    canvas = Image.new('RGB', (1200, 900), 'white'); draw = ImageDraw.Draw(canvas)
    draw.text((12, 5), f"{pair['display_index']:03d}  pair={pair['pair_id']}  {pair['kind']}  待两人独立复核", font=font(), fill='black')
    detail = f"Trans10K split={b['split']}"
    if 'phash_distance' in pair:
        detail += f" | pHash={pair['phash_distance']}, dHash={pair['dhash_distance']} | {pair['method']}"
    draw.text((12, 33), detail, font=font(), fill='black')
    for col, (row, base, encoding, name) in enumerate(((a, DATA, 'binary_0255', 'GSD'),
            (b, b['base'], 'trans10k_v1_rgb', 'Trans10K'))):
        views = load_views(row, base, encoding)
        for line, text in enumerate(textwrap.wrap(name+': '+row['id'], 57)):
            draw.text((col*600+12, 65+line*22), text, font=font(17), fill='black')
        paste_fit(canvas, views[0], (col*600+10, 118, 580, 365))
        draw.text((col*600+12, 494), '各自标签叠图：绿色为前景；标签范围可能不同', font=font(17), fill='black')
        paste_fit(canvas, views[2], (col*600+10, 523, 580, 365))
    return canvas


def validate_decisions(review, template):
    for key in ('schema', 'audit_binding_sha256', 'policy_sha256', 'full_manifest_sha256'):
        if review.get(key) != template[key]:
            raise ValueError('Review does not bind to this audit: '+key)
    pairs = review.get('pairs', [])
    originals = {r['pair_id']: r for r in template['pairs']}
    if len(pairs) != len(originals) or {r['pair_id'] for r in pairs} != set(originals):
        raise ValueError('Missing/duplicate/unrecognized review pairs')
    decisions = []
    for r in pairs:
        original = originals[r['pair_id']]
        for key, value in original.items():
            if key not in ('member4', 'member1') and r.get(key) != value:
                raise ValueError('Candidate evidence changed: '+r['pair_id'])
        people = []
        for role in ('member4','member1'):
            vote = r.get(role, {})
            if (vote.get('decision') not in template['allowed_decisions']
                    or not str(vote.get('reviewer', '')).strip() or not str(vote.get('reason','')).strip()):
                raise ValueError('Both reviewers must explicitly record a decision/name/reason: '+r['pair_id'])
            people.append(vote['reviewer'].strip().casefold())
        if people[0] == people[1]:
            raise ValueError('Two independent reviewers are required')
        x, y = r['member4']['decision'], r['member1']['decision']
        if r['kind'] != 'perceptual_candidate':
            if x != 'same_source' or y != 'same_source':
                raise ValueError('An exact decoded-image match cannot be silently marked different; investigate the evidence')
            exclude, reason = True, 'exact_match_reviewed'
        elif x == y == 'same_source':
            exclude, reason = True, 'same_source_reviewed'
        elif x == y and x in ('not_same_source','different_capture_same_scene'):
            exclude, reason = False, x
        else:
            exclude, reason = True, 'unresolved_or_disagreement'
        decisions.append({'pair_id':r['pair_id'],'gsd_id':r['gsd_id'], 'trans10k_id':r['trans10k_id'],
            'trans10k_split':r['trans10k_split'], 'exclude_from_supplementary':exclude, 'reason':reason,
            'member4':r['member4'],'member1':r['member1']})
    for key, identity, allowed in (('labels','gsd_id',('pass','question','problem')),
            ('internal_duplicate_issues','issue_id',('acknowledged_keep','question'))):
        expected={r[identity] for r in template[key]}; entries=review.get(key,[])
        if len(entries)!=len(expected) or {r[identity] for r in entries}!=expected:
            raise ValueError('Incomplete '+key)
        if any(r.get('status') not in allowed or not r.get('reviewer','').strip() or not r.get('reason','').strip() for r in entries):
            raise ValueError('All label/internal issues need explicit reviewer, status and reason')
    member5=review.get('member5',{})
    if not member5.get('reviewer','').strip() or not member5.get('reason','').strip():
        raise ValueError('Member5 policy review must be explicitly recorded')
    for key in ('policy_approved','target_semantics_acknowledged','limitations_acknowledged','internal_duplicate_policy_approved'):
        if member5.get(key) is not True:
            raise ValueError('Member5 confirmation missing: '+key)
    if review.get('user_stage5_accepted') is not True:
        raise ValueError('User stage5 acceptance remains pending')
    audit_people=[]
    for role in ('member4','member1'):
        entry=review.get('cross_audit_review',{}).get(role,{})
        if entry.get('status')!='reviewed' or not entry.get('reviewer','').strip() or not entry.get('reason','').strip():
            raise ValueError('Cross-dataset retrieval and diagnostic images still need both reviewers; questions require investigation')
        audit_people.append(entry['reviewer'].strip().casefold())
    if audit_people[0]==audit_people[1]:
        raise ValueError('Cross audit requires two independent reviewers')
    return decisions


def freeze_review(review_path, output):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Freeze into a new directory; do not overwrite prior review versions')
    template, review = read(ROOT/'human_review_template.json'), read(review_path)
    bindings = {'audit_binding_sha256': ROOT/'cross_candidates.json',
                'policy_sha256':ROOT/'policy_before_retrieval.json', 'full_manifest_sha256':MANIFESTS/'test_full.json'}
    if any(sha256_file(p)!=template[k] for k,p in bindings.items()):
        raise ValueError('Underlying audit or manifest changed')
    decisions = validate_decisions(review, template)
    full, base=read_manifest(MANIFESTS/'test_full.json')
    excluded={d['gsd_id'] for d in decisions if d['exclude_from_supplementary']}
    retained=[r for r in full['samples'] if r['id'] not in excluded]
    if not retained:
        raise ValueError('No supplementary samples remain; do not create an unevaluable empty manifest')
    output.mkdir(parents=True)
    full={**full,'root':Path(os.path.relpath(base,output.resolve())).as_posix(),
        'review_status':'human_review_recorded', 'review_decisions_sha256':sha256_file(review_path),
        'evaluation_authorized':False, 'next_gate':'Explicit stage6 authorization is still required'}
    full['data_protocol']={**full['data_protocol'],'status':'human_review_recorded'}
    subset={**full, 'complete_pinned_source_split':False, 'effective_count':len(retained),'samples':retained,
            'subset_status':'reviewed', 'subset_limitations':'Finite retrieval cannot prove no remaining overlap; internal duplicate members retained.'}
    write_json(output/'test_full.json',full); write_json(output/'test_supplementary.json',subset)
    write_json(output/'pair_decisions.json',decisions); write_json(output/'human_review_record.json',review)
    write_json(output/'exclusions.json',[{'gsd_id':r['id'], 'reasons':[d for d in decisions if d['gsd_id']==r['id'] and d['exclude_from_supplementary']]} for r in full['samples'] if r['id'] in excluded])
    write_json(output/'same_scene_different_capture.json',[d for d in decisions if d['reason']=='different_capture_same_scene'])
    read_manifest(output/'test_full.json'); read_manifest(output/'test_supplementary.json')
    write_json(output/'release.json',{'full_count':813, 'supplementary_count':len(retained),
        'unresolved_pairs_excluded':sum(d['reason']=='unresolved_or_disagreement' for d in decisions),
        'stage6_started':False, 'manifest_sha256':{p.name:sha256_file(p) for p in output.glob('test*.json')}})
    print('Human-reviewed manifests frozen; stage6 has not started.',flush=True)


def render():
    target=ROOT/'previews'
    if target.exists():
        raise FileExistsError('Review previews already exist; preserve prior evidence')
    (target/'labels').mkdir(parents=True); (target/'cross').mkdir(); (target/'contacts').mkdir(); (target/'diagnostics').mkdir()
    g={r['id']:r for r in read(ROOT/'gsd_inventory.json')}
    t={r['id']:r for r in read(ROOT/'trans10k_inventory.json')}
    review=read(ROOT/'human_review_template.json')
    label_paths=[]
    for n,r in enumerate(review['labels'],1):
        path=target/'labels'/f'{n:02d}.png'; label_panel(g[r['gsd_id']]).save(path)
        label_paths.append(path)
    for page in range(4):
        sheet=Image.new('RGB',(1200,1250),'white')
        for slot,path in enumerate(label_paths[page*5:(page+1)*5]):
            with Image.open(path) as im: paste_fit(sheet,im,(0,slot*250,1200,250))
        sheet.save(target/f'labels_contact_{page+1:02d}.png')
    pair_index=[]
    for n,pair in enumerate(review['pairs'],1):
        panel=cross_panel({**pair,'display_index':n},g[pair['gsd_id']],t[pair['trans10k_id']])
        path=target/'cross'/f'{n:04d}.png'; panel.save(path)
        pair_index.append({'index':n, 'pair_id':pair['pair_id'], 'file':path.relative_to(ROOT).as_posix(),
                          'kind':pair['kind'],'gsd_id':pair['gsd_id'],'trans10k_id':pair['trans10k_id'],
                          'trans10k_split':pair['trans10k_split']})
        if n%25==0:
            print(f'Review pair panels: {n}/{len(review["pairs"])}',flush=True)
    for page in range((len(pair_index)+7)//8):
        sheet=Image.new('RGB',(1200,1200),'white'); draw=ImageDraw.Draw(sheet)
        for slot,r in enumerate(pair_index[page*8:(page+1)*8]):
            x,y=(slot%2)*600,(slot//2)*300
            draw.text((x+5,y+4),f"{r['index']:03d} | {r['trans10k_split']} | {r['kind']}",font=font(15),fill='black')
            for col,(row,base) in enumerate(((g[r['gsd_id']],DATA),(t[r['trans10k_id']],t[r['trans10k_id']]['base']))):
                with Image.open(Path(base)/row['image']) as im: paste_fit(sheet,im.convert('RGB'),(x+col*300+4,y+30,292,260))
        sheet.save(target/'contacts'/f'{page+1:03d}.png')
    duplicates=read(ROOT/'within_gsd_duplicates.json')
    diagnostics=read(ROOT/'retrieval_diagnostics.json')['diagnostic_pairs']
    for n,pair in enumerate(diagnostics,1):
        cross_panel({**pair,'display_index':n},g[pair['gsd_id']],t[pair['trans10k_id']]).save(target/'diagnostics'/f'{n:02d}.png')
    for page in range((len(diagnostics)+5)//6):
        canvas=Image.new('RGB',(1200,1020),'white'); draw=ImageDraw.Draw(canvas)
        for slot,pair in enumerate(diagnostics[page*6:(page+1)*6]):
            x,y=(slot%2)*600,(slot//2)*340
            draw.text((x+5,y+5),f"诊断 {page*6+slot+1:02d} | {pair['trans10k_split']} | p={pair['phash_distance']}, d={pair['dhash_distance']}",font=font(16),fill='black')
            draw.text((x+5,y+31),'未达到候选阈值；左GSD / 右Trans10K',font=font(16),fill='black')
            for col,(row,base) in enumerate(((g[pair['gsd_id']],DATA),(t[pair['trans10k_id']],t[pair['trans10k_id']]['base']))):
                with Image.open(Path(base)/row['image']) as im:
                    paste_fit(canvas,im.convert('RGB'),(x+col*300+4,y+60,292,270))
        canvas.save(target/f'diagnostics_contact_{page+1:02d}.png')
    for issue in duplicates:
        a,b=[g[i] for i in issue['ids'][:2]]
        va,vb=load_views(a,DATA,'binary_0255'),load_views(b,DATA,'binary_0255')
        ma=np.asarray(va[1])[:,:,0]; mb=np.asarray(vb[1])[:,:,0]
        diff=np.zeros((*ma.shape,3),np.uint8); diff[ma!=mb]=[255,60,60]
        canvas=Image.new('RGB',(1600,500),'white'); draw=ImageDraw.Draw(canvas)
        draw.text((12,5),f"{issue['issue_id']} | 相同原图、不同标签 | 差异像素={issue['comparisons'][0]['mask_different_pixels']}",font=font(),fill='black')
        for col,(view,label) in enumerate(zip((va[0],va[1],vb[1],Image.fromarray(diff)),('共同原图',a['source_id'],b['source_id'],'红色=标签不同'))):
            draw.text((col*400+8,40),label,font=font(15),fill='black'); paste_fit(canvas,view,(col*400+5,80,390,410))
        canvas.save(target/(issue['issue_id']+'.png'))
    write_json(ROOT/'preview_index.json',{'pairs':pair_index,'diagnostic_pairs':[{'index':n+1,**r,'file':f'previews/diagnostics/{n+1:02d}.png'} for n,r in enumerate(diagnostics)],'labels':[{'index':n+1,'gsd_id':r['gsd_id'],'file':label_paths[n].relative_to(ROOT).as_posix()} for n,r in enumerate(review['labels'])]})
    lines=['GSD 阶段5待人工复核表','全部条目尚未由成员签字；可在review.html填写并导出JSON。','',
        '判定值：same_source=同源照片（含缩放/重压缩/裁剪）；different_capture_same_scene=同场景不同拍摄；not_same_source=不同来源；uncertain=无法确定。',
        '每对由成员4、成员1独立填写姓名、判定、理由；分歧或无法确定者继续排除于补充子集。','']
    if not pair_index:
        lines += ['跨数据集精确重复及阈值内近重复候选均为0；因此没有逐候选判定行。',
                  '请成员4和成员1查看12组最近邻诊断图并复核检索范围/限制；这些图不是达到阈值的候选，不构成自动排除理由。',
                  '若人工发现疑似同源或无法判断的图片，请记录GSD ID、Trans10K ID和理由，先交回调查，再冻结清单。','']
    lines += [f"标签抽样：20项待填写；GSD内部同图异标：{len(duplicates)}项待复核；成员5规则复核：待填写。",'']
    for r in pair_index:
        lines += [f"{r['index']:03d} | {r['pair_id']} | {r['kind']} | Trans10K {r['trans10k_split']}",
            f"GSD: {r['gsd_id']}",f"Trans10K: {r['trans10k_id']}",f"图片: {r['file']}",
            '成员4：待复核；姓名____ 判定____ 理由____','成员1：待复核；姓名____ 判定____ 理由____','']
    (ROOT/'待人工复核表.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    build_html(review,pair_index,duplicates)
    print(f"Rendered 20 annotation panels, {len(pair_index)} cross-dataset pairs and {len(duplicates)} internal issue panels.",flush=True)


def build_html(review,pairs,issues):
    # Inline data + local images only; there is no server, upload, or network request.
    payload=json.dumps(review,ensure_ascii=False).replace('<','\\u003c')
    index=json.dumps(pairs,ensure_ascii=False).replace('<','\\u003c')
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GSD 阶段5人工复核</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1280px;margin:24px auto;padding:0 20px;color:#24323d;background:#f6f8fb}h1,h2{line-height:1.3}section{background:white;border:1px solid #d9e0e6;border-radius:8px;padding:18px;margin:16px 0}img{max-width:100%;height:auto}input[type=text],textarea{box-sizing:border-box;width:100%;padding:8px}textarea{min-height:58px}select,button{font:inherit;padding:8px;margin:4px}label{display:block;margin:7px 0}.votes{display:grid;grid-template-columns:1fr 1fr;gap:20px}.bar{position:sticky;top:0;background:#fff6dc;border:1px solid #ead294;padding:12px;z-index:1}small{color:#53616b}.tag{font-weight:bold;color:#8b3f13}a{color:#125a8f}summary{cursor:pointer;font-weight:bold}code{overflow-wrap:anywhere}</style>
<h1>GSD 数据与跨数据集候选复核</h1><p>完整813张保留原样；所有人工意见初始为空。此页面只保存你的复核记录，不会运行模型或修改图片。候选不等于已确认同源。</p>
<p><a href="audit.json">机器检查</a> · <a href="待人工复核表.txt">TXT表</a> · <a href="policy_before_retrieval.json">检索前固定规则</a></p>
<div class="bar"><button id="download">导出已填写的复核JSON</button><label style="display:inline">导入先前记录 <input id="import" type="file" accept=".json"></label><span id="status"></span><br><small>不会自动保存；关闭前请导出。未解决/意见不一致者仍列为排除候选。不要代填他人的签字。</small></div>
<h2>1. 固定20张标签：成员4填写，疑点交成员1复核</h2><p>检查白色是否对应玻璃、位置是否对齐、边框和遮挡范围是否合理。不同任务的标注范围可能不同；本次不重画标签。</p><div id="labels"></div>
<h2>2. 跨数据集候选：成员4与成员1独立判定</h2><p>同源照片包括重压缩、缩放或裁剪版本；相同场景的另一张照片需单独标记。精确字节/像素匹配有确定的机器证据，若认为不匹配，请先排查证据。</p>
<label>按Trans10K划分查看 <select id="filter"><option value="all">全部</option><option>train</option><option>validation</option><option>test</option></select></label><div id="pairs"></div>
<h2>跨库最近邻诊断（未达到候选阈值）</h2><p>以下12组仅用于检查检索结果。不是确认同源，不构成自动排除理由；有限检索不能证明无重叠。若发现疑点，请选“存疑”并记录具体ID，交回调查后再冻结清单。</p><div id="diagnostics"></div><section id="cross_audit"></section>
<h2>3. GSD内部重复与标签差异</h2><div id="internal"></div>
<h2>4. 成员5复核处理规则</h2><section id="policy"></section>
<h2>5. 用户阶段5验收</h2><label><input id="accept" type="checkbox">我已查看材料并确认阶段5通过（不会自动启动阶段6）</label>
<p>导出的JSON请保存在项目中，告知文件位置。冻结操作会验证候选、清单和规则哈希，以及所有必填项；只有材料准备完成并不等于已经获得组员确认。</p>
<script>
let data=__DATA__;const template=structuredClone(data), pairIndex=__INDEX__, diagnosticPairs=__DIAGNOSTICS__;
const e=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
const options={pending:'待复核',same_source:'同源照片',different_capture_same_scene:'同场景不同拍摄',not_same_source:'不同来源',uncertain:'无法确定',pass:'通过',question:'存疑',problem:'明显问题',acknowledged_keep:'确认保留原成员并披露差异',reviewed:'已查看诊断图并复核检索范围/限制'};
function bind(parent,obj,key,label,type='text',choices=[]){const l=e('label',label+' '),i=e(type==='select'?'select':type==='textarea'?'textarea':'input');if(type==='select'){for(const v of choices){const o=e('option',options[v]||v);o.value=v;i.append(o);}i.value=obj[key];}else if(type==='checkbox'){i.type='checkbox';i.checked=!!obj[key];}else{i.type='text';i.value=obj[key]||'';}i.addEventListener('input',()=>{obj[key]=type==='checkbox'?i.checked:i.value;});l.append(i);parent.append(l);}
function picture(parent,path){const a=e('a');a.href=path;a.target='_blank';const im=e('img');im.src=path;im.loading='lazy';a.append(im);parent.append(a);}
function render(){for(const id of ['labels','pairs','internal','policy','diagnostics','cross_audit'])document.getElementById(id).replaceChildren();
data.labels.forEach((r,i)=>{const s=e('section'),d=e('details');d.append(e('summary',String(i+1).padStart(2,'0')+' '+r.gsd_id));picture(d,'previews/labels/'+String(i+1).padStart(2,'0')+'.png');s.append(d);bind(s,r,'status','标签检查','select',['pending','pass','question','problem']);bind(s,r,'reviewer','检查人');bind(s,r,'reason','观察/理由','textarea');document.getElementById('labels').append(s);});
data.pairs.forEach((r,i)=>{const s=e('section');s.dataset.split=r.trans10k_split;const d=e('details');d.append(e('summary',String(i+1).padStart(3,'0')+' | '+r.trans10k_split+' | '+r.kind+' | '+r.pair_id));d.append(e('p','GSD: '+r.gsd_id+' / Trans10K: '+r.trans10k_id));picture(d,pairIndex[i].file);s.append(d);const votes=e('div');votes.className='votes';for(const [role,label]of [['member4','成员4'],['member1','成员1']]){const box=e('div');box.append(e('strong',label));bind(box,r[role],'reviewer','姓名');bind(box,r[role],'decision','判定','select',['pending',...data.allowed_decisions]);bind(box,r[role],'reason','理由','textarea');votes.append(box);}s.append(votes);document.getElementById('pairs').append(s);});
if(!data.pairs.length)document.getElementById('pairs').append(e('p','本次阈值内跨数据集候选为0；不代表完全没有重叠。'));
diagnosticPairs.forEach((r,i)=>{const s=e('section'),d=e('details');d.append(e('summary','诊断 '+String(i+1).padStart(2,'0')+' | '+r.trans10k_split+' | pHash='+r.phash_distance+' / dHash='+r.dhash_distance));picture(d,'previews/diagnostics/'+String(i+1).padStart(2,'0')+'.png');s.append(d);document.getElementById('diagnostics').append(s);});
for(const[role,label]of [['member4','成员4'],['member1','成员1']]){const s=e('section');s.append(e('strong',label+'：跨库检查'));bind(s,data.cross_audit_review[role],'reviewer','姓名');bind(s,data.cross_audit_review[role],'status','意见','select',['pending','reviewed','question']);bind(s,data.cross_audit_review[role],'reason','复核说明/疑似图片ID','textarea');document.getElementById('cross_audit').append(s);}
data.internal_duplicate_issues.forEach(r=>{const s=e('section');s.append(e('strong',r.issue_id));picture(s,'previews/'+r.issue_id+'.png');bind(s,r,'status','意见','select',['pending','acknowledged_keep','question']);bind(s,r,'reviewer','复核人');bind(s,r,'reason','处理意见（完整集始终保留原样）','textarea');document.getElementById('internal').append(s);});
const p=document.getElementById('policy');bind(p,data.member5,'reviewer','成员5姓名');for(const[k,l]of [['policy_approved','认可排除所有Trans10K划分中的确认同源及未解决候选'],['target_semantics_acknowledged','已检查GSD玻璃与Trans10K透明物体标注范围差异'],['limitations_acknowledged','认可有限检索的漏检风险及已记录的标签疑点'],['internal_duplicate_policy_approved','认可完整集及补充集保留GSD内部重复成员和原标签并披露差异']])bind(p,data.member5,k,l,'checkbox');bind(p,data.member5,'reason','规则复核意见','textarea');document.getElementById('accept').checked=!!data.user_stage5_accepted;filter();}
function filter(){const v=document.getElementById('filter').value;for(const s of document.getElementById('pairs').children)s.hidden=v!=='all'&&s.dataset.split!==v;}
document.getElementById('filter').onchange=filter;document.getElementById('accept').onchange=ev=>data.user_stage5_accepted=ev.target.checked;
document.getElementById('download').onclick=()=>{const b=new Blob([JSON.stringify(data,null,2)+'\\n'],{type:'application/json'}),u=URL.createObjectURL(b),a=e('a');a.href=u;a.download='gsd_stage5_review_decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);document.getElementById('status').textContent='已导出当前记录；未填项目仍为待复核。';};
document.getElementById('import').onchange=async ev=>{try{const x=JSON.parse(await ev.target.files[0].text());if(x.audit_binding_sha256!==template.audit_binding_sha256||x.pairs.length!==template.pairs.length||x.pairs.some((r,i)=>r.pair_id!==template.pairs[i].pair_id))throw Error('不是本次审计或候选顺序不一致');data=x;render();document.getElementById('status').textContent='已导入记录';}catch(err){document.getElementById('status').textContent='导入失败：'+err.message;}};render();
</script></html>'''
    diagnostic_json=json.dumps(read(ROOT/'retrieval_diagnostics.json')['diagnostic_pairs'],ensure_ascii=False).replace('<','\\u003c')
    (ROOT/'review.html').write_text(document.replace('__DATA__',payload).replace('__INDEX__',index).replace('__DIAGNOSTICS__',diagnostic_json),encoding='utf-8',newline='\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('render')
    p=sub.add_parser('freeze'); p.add_argument('--review',required=True); p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='render': render()
    else: freeze_review(args.review,args.output)

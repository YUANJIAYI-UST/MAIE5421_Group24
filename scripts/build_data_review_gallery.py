"""Build a portable, offline review gallery; human decisions are exported explicitly."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transparent_seg.data import decode_mask
from transparent_seg.prepare import sha256_file, write_json
from scripts.visualize_trans10k_audit import font, make_panel


HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>成员 1 数据复核</title><style>
body{font:16px/1.6 system-ui,sans-serif;margin:0;background:#f4f6f9;color:#172334}header,main{max-width:1100px;margin:auto;padding:24px}header{padding-bottom:8px}h1{margin:0;font-size:28px}button,select,input,textarea{font:inherit;padding:8px;border:1px solid #bac5d0;border-radius:6px}button{background:#174b78;color:white;cursor:pointer}button:disabled{opacity:.4}nav{display:flex;gap:12px;align-items:center;flex-wrap:wrap}article{background:white;border:1px solid #dde2e8;border-radius:10px;margin:20px 0;padding:18px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.pair img{width:100%;height:300px;object-fit:contain;background:#eee}img.geometry{width:100%;height:auto}.caption{font-size:13px;overflow-wrap:anywhere}.status{padding:12px;background:#fff1d0;border-radius:6px}textarea{display:block;width:calc(100% - 18px);margin-top:8px}label{display:block;margin-top:12px}#progress{font-weight:600}.small{font-size:14px;color:#536170}@media(max-width:650px){.pair{grid-template-columns:1fr}.pair img{height:240px}}
</style><header><h1>数据复核：方向修复与相似图片</h1>
<p>18 张标注已在副本中旋转；已确认的跨划分完全重复项已从派生清单排除。下面的相似图片均为候选，尚未因相似度自动排除。</p>
<p class="status">此页面只收集复核意见。保存／导出不会修改数据。最终清单需根据意见重新生成并校验。AI 检查与同学人工复核分开记录。</p>
<label>复核者姓名（导出已判断项目时必填） <input id="reviewer" placeholder="填写你自己的姓名"></label>
<nav><select id="filter"><option value="cross">跨划分相似候选（优先）</option><option value="geometry">18 张方向修复</option><option value="within">同划分相似候选</option><option value="all">全部项目</option></select>
<button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button><button id="export">导出复核 JSON</button></nav>
<p id="progress"></p><p class="small">每页 8 项。选择和理由会在切换页面时保留；关闭／刷新前请导出。不能确认时选“不确定”。“同一场景的不同照片”会单独记录，不会直接等同于重复照片。</p><p id="message" role="status"></p></header><main id="items"></main>
<script id="payload" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent);
const decisions={};let currentPage=0;const pageSize=8;
const items=document.getElementById('items');
function selected(){const filter=document.getElementById('filter').value;return data.items.filter(x=>filter==='all'||(filter==='geometry'?x.kind==='geometry':x.kind==='near'&&(filter==='cross'?x.cross_split:!x.cross_split)));}
function progress(){const values=Object.values(decisions);document.getElementById('progress').textContent=`共 ${data.items.length} 项；已填写 ${values.filter(x=>x.decision!=='pending').length} 项（尚未代表全部通过）`;}
function render(){const list=selected(),pages=Math.max(1,Math.ceil(list.length/pageSize));currentPage=Math.min(currentPage,pages-1);items.replaceChildren();document.getElementById('page').textContent=`${currentPage+1} / ${pages} 页 · ${list.length} 项`;document.getElementById('prev').disabled=currentPage===0;document.getElementById('next').disabled=currentPage===pages-1;
for(const x of list.slice(currentPage*pageSize,(currentPage+1)*pageSize)){const a=document.createElement('article');const h=document.createElement('h2');h.style.fontSize='18px';h.textContent=x.title;a.append(h);
if(x.kind==='geometry'){const img=document.createElement('img');img.className='geometry';img.src=x.panel;img.alt=x.title;a.append(img);}else{const pair=document.createElement('div');pair.className='pair';x.ids.forEach(id=>{const cell=document.createElement('div'),img=document.createElement('img'),caption=document.createElement('div');img.src=data.thumbnails[id];img.alt=id;caption.className='caption';caption.textContent=id;cell.append(img,caption);pair.append(cell);});a.append(pair);}
const evidence=document.createElement('p');evidence.className='small';evidence.textContent=x.evidence;a.append(evidence);
const options=x.kind==='geometry'?[['pending','待复核'],['aligned','修复后对齐'],['incorrect','仍不正确'],['uncertain','不确定']]:[['pending','待复核'],['same_source_photo','同一原照片的裁剪／缩放／压缩版本'],['same_scene','同一场景的不同照片'],['distinct','不同照片，仅外观相似'],['uncertain','不确定']];
const select=document.createElement('select');select.setAttribute('aria-label','判断 '+x.key);for(const [value,label] of options){const o=document.createElement('option');o.value=value;o.textContent=label;select.append(o);}select.value=decisions[x.key]?.decision||'pending';const reason=document.createElement('textarea');reason.rows=2;reason.placeholder='填写判断理由，指出相同细节或差异';reason.setAttribute('aria-label','理由 '+x.key);reason.value=decisions[x.key]?.reason||'';
function update(){decisions[x.key]={decision:select.value,reason:reason.value.trim()};progress();}select.addEventListener('change',update);reason.addEventListener('input',update);a.append(select,reason);items.append(a);}progress();}
document.getElementById('filter').addEventListener('change',()=>{currentPage=0;render();});document.getElementById('prev').onclick=()=>{currentPage--;render();};document.getElementById('next').onclick=()=>{currentPage++;render();};
document.getElementById('export').onclick=()=>{const reviewer=document.getElementById('reviewer').value.trim();const judged=Object.values(decisions).filter(x=>x.decision!=='pending');const msg=document.getElementById('message');if(judged.length&&(!reviewer||judged.some(x=>!x.reason))){msg.textContent='请填写复核者姓名，并为每个已判断项目填写理由。';return;}const records=data.items.map(x=>({key:x.key,kind:x.kind,ids:x.ids,source_image_sha256:x.source_image_sha256,decision:decisions[x.key]?.decision||'pending',reason:decisions[x.key]?.reason||'',reviewer:decisions[x.key]?.decision&&decisions[x.key].decision!=='pending'?reviewer:null}));const result={schema:'trans10k_human_review_v1',candidate_sha256:data.candidate_sha256,policy_sha256:data.policy_sha256,reviewer:reviewer||null,reviewed_at:new Date().toISOString(),data_modified:false,records};const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='trans10k_human_review.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);msg.textContent='已导出复核文件。请保存并交给数据负责人，数据清单尚未改变。';};render();
</script></html>'''


def jpeg_uri(image):
    buffer=BytesIO()
    image.save(buffer, format='JPEG', quality=88)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()


def build(root, candidate_path, manifest_dir, output):
    root, manifest_dir, output = Path(root).resolve(), Path(manifest_dir), Path(output)
    if output.exists():
        raise FileExistsError('Use a new gallery directory')
    output.mkdir(parents=True)
    candidates=json.loads(Path(candidate_path).read_text())
    audit=json.loads((manifest_dir/'audit.json').read_text())
    items, thumbnails, panels=[], {}, []
    for correction in audit['corrections']:
        sid=correction['id']
        with Image.open(root/sid) as im: image=im.convert('RGB')
        y,valid=decode_mask(root.parent/correction['output_mask'])
        assert image.size==(y.shape[1],y.shape[0]) and valid.all()
        display=image.copy();display.thumbnail((500,350),Image.Resampling.LANCZOS)
        fg=np.asarray(Image.fromarray(y.astype(np.uint8)).resize(display.size,Image.Resampling.NEAREST)).astype(bool)
        rgb=np.array(display).astype(float);rgb[fg]=0.55*rgb[fg]+0.45*np.array([0,220,160])
        panel=make_panel(image,Image.fromarray(y.astype(np.uint8)*255).convert('RGB'),Image.fromarray(rgb.astype(np.uint8)),sid+' | applied to copy: '+correction['operation'])
        filename=sid.replace('/','_')+'.jpg';panel.save(output/filename,quality=94);panels.append(panel)
        items.append({'key':'geometry:'+sid,'kind':'geometry','ids':[sid],
                      'source_image_sha256':[sha256_file(root/sid)], 'title':'方向修复 · '+sid,
                      'evidence':correction['operation']+'；原标注保留，副本已旋转；需确认轮廓与照片对齐。',
                      'panel':jpeg_uri(panel)})
    for start in range(0,len(panels),4):
        subset=panels[start:start+4];sheet=Image.new('RGB',(1232,366*len(subset)),'white')
        for i,panel in enumerate(subset):sheet.paste(panel,(0,i*366))
        sheet.save(output/f'repaired_contact_{start//4+1:02d}.jpg',quality=94)
    for candidate in candidates['candidates']:
        for sid in candidate['ids']:
            if sid not in thumbnails:
                with Image.open(root/sid) as im:
                    thumb=im.convert('RGB');thumb.thumbnail((420,300),Image.Resampling.LANCZOS)
                thumbnails[sid]=jpeg_uri(thumb)
        items.append({'key':candidate['pair_id'],'kind':'near', 'ids':candidate['ids'],
                      'source_image_sha256':candidate['source_image_sha256'], 'cross_split':candidate['cross_split'],
                      'title':('跨划分' if candidate['cross_split'] else '同划分')+'相似候选 · '+candidate['pair_id'],
                      'evidence':f"检索依据：{candidate['method']}；pHash 距离 {candidate['phash_distance']}；dHash 距离 {candidate['dhash_distance']}。尚未确认是否重复。"})
    payload={'candidate_sha256':sha256_file(candidate_path),'policy_sha256':audit['data_protocol']['policy_sha256'],
             'items':items,'thumbnails':thumbnails}
    text=HTML.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'))
    (output/'review.html').write_text(text,encoding='utf-8')
    template={'schema':'trans10k_human_review_v1','candidate_sha256':payload['candidate_sha256'],
              'policy_sha256':payload['policy_sha256'],'reviewer':None,'data_modified':False,
              'records':[{k:x[k] for k in ('key','kind','ids','source_image_sha256')} | {'decision':'pending','reason':'','reviewer':None} for x in items]}
    write_json(output/'human_review_template.json',template)
    # Compact contact sheets are useful for AI triage; they do not fill in human decisions.
    for cross in (True,False):
        pairs=[c for c in candidates['candidates'] if c['cross_split']==cross]
        for start in range(0,len(pairs),6):
            subset=pairs[start:start+6];sheet=Image.new('RGB',(1000,280*len(subset)),'#f1f5f9');draw=ImageDraw.Draw(sheet)
            for i,c in enumerate(subset):
                draw.text((10,i*280+4),f"{start+i+1:03d} | {c['pair_id']} | pH {c['phash_distance']} dH {c['dhash_distance']}",fill='black',font=font(17))
                for j,sid in enumerate(c['ids']):
                    with Image.open(root/sid) as im:thumb=im.convert('RGB');thumb.thumbnail((480,218),Image.Resampling.LANCZOS)
                    sheet.paste(thumb,(10+j*500+(480-thumb.width)//2,i*280+32+(218-thumb.height)//2))
                    draw.text((10+j*500,i*280+253),sid,fill='black',font=font(15))
            sheet.save(output/f'near_{"cross" if cross else "within"}_contact_{start//6+1:02d}.jpg',quality=91)
    print(json.dumps({'items':len(items),'geometry':len(audit['corrections']),'near_candidates':len(candidates['candidates']),
                      'human_review':'pending','gallery':str(output/'review.html')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='data/Trans10K');p.add_argument('--candidates',required=True)
    p.add_argument('--manifests',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();build(a.root,a.candidates,a.manifests,a.output)

"""Create static D03 review panels; all human decisions remain pending."""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transparent_seg.data import decode_mask
from transparent_seg.data_v2 import CLASS_NAMES
from transparent_seg.prepare import write_json


def transformed(im,operation):
    if operation=='rotate_90_cw':return im.transpose(Image.Transpose.ROTATE_270)
    if operation=='rotate_90_ccw':return im.transpose(Image.Transpose.ROTATE_90)
    if operation=='none':return im.copy()
    raise ValueError(operation)


def make_panel(row,data,base_lookup):
    font=ImageFont.load_default(size=17)
    sample=base_lookup.get(row['id'])
    with Image.open(data/'Trans10K'/row['id']) as im: raw=im.convert('RGB')
    with Image.open(data/'Trans10K_v2_mirror_3445e93'/row['v2_image']) as im:
        v2=transformed(im.convert('RGB'),row['v2_to_v1_orientation'])
    with Image.open(data/'Trans10K_v2_mirror_3445e93'/row['v2_mask']) as im:
        fine=transformed(im.convert('RGB'),row['v2_to_v1_orientation'])
    tiles=[]
    for im in (raw,v2,fine):
        im.thumbnail((350,270),Image.Resampling.NEAREST if im is fine else Image.Resampling.LANCZOS)
        tile=Image.new('RGB',(360,280),'white');tile.paste(im,((360-im.width)//2,(280-im.height)//2));tiles.append(tile)
    panel=Image.new('RGB',(1080,340),'white')
    draw=ImageDraw.Draw(panel)
    draw.text((8,3),row['id']+' | classes '+','.join(str(x) for x in row['category_ids'])+
              ' | '+row['v2_to_v1_orientation'],fill='black',font=font)
    for j,(tile,label) in enumerate(zip(tiles,['Original V1 image','Aligned V2 image','V2 category colors (not V1 GT)'])):
        panel.paste(tile,(j*360,55));draw.text((j*360+8,32),label,fill='black',font=font)
    return panel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/trans10k_d03_20261028')
    p.add_argument('--base',default='manifests/trans10k_v1_review_20261028')
    p.add_argument('--data',default='data')
    args=p.parse_args();run=Path(args.run);out=run/'human_review'
    if out.exists():raise FileExistsError('Use a new review output')
    rows=[]
    for path in ('train_val_aligned/inventory.json','test/inventory.json'):
        rows+=json.loads((run/path).read_text())['samples']
    selection=json.loads((run/'selection/selection.json').read_text())
    lookup={}
    for split in ('train','validation','test'):
        lookup.update({s['id']:s for s in json.loads((Path(args.base)/f'{split}.json').read_text())['samples']})
    groups={}
    for c in selection['held_out_class_ids']:
        picked=[]
        for split in ('train','validation'):
            candidates=[r for r in rows if r['split']==split and c in r['category_ids'] and r['category_filter_eligible']]
            picked+=random.Random(20261028+c).sample(candidates,min(3,len(candidates)))
        groups[f'class_{c:02d}']=picked
    groups['documented_rotations']=[r for r in rows if r['v2_to_v1_orientation']!='none']
    groups['quarantined']=[r for r in rows if r['issues']]
    out.mkdir(parents=True)
    items=[]
    for group,members in groups.items():
        panels=[]
        for n,row in enumerate(members):
            panel=make_panel(row,Path(args.data),lookup)
            filename=f'{group}_{n:02d}.jpg';panel.save(out/filename,quality=90)
            panels.append(panel)
            items.append({'id':row['id'],'group':group,'panel':filename,'classes':row['category_ids'],
                          'issues':row['issues'],'unknown_codes':row['unknown_codes'],
                          'human_reviewer':None,'decision':'pending','reason':''})
        for page,start in enumerate(range(0,len(panels),6)):
            chunk=panels[start:start+6];sheet=Image.new('RGB',(1080,340*len(chunk)),'white')
            for j,panel in enumerate(chunk):sheet.paste(panel,(0,j*340))
            sheet.save(out/f'{group}_contact_{page+1:02d}.jpg',quality=88)
    write_json(out/'review_template.json',{'status':'pending_human_review','source':'AI-prepared review panels; not human decisions',
               'protocol_decisions':[{'item':'Use this pinned third-party mirror or obtain and compare the author Baidu archive; mirror card mentions an upstream red-to-cup color correction','decision':'pending'},
                                     {'item':'Approve train/validation-only selection of class IDs 5 and 8; resolve class 8 name discrepancy','decision':'pending'}],
               'items':items})
    print(json.dumps({'panels':len(items),'groups':{k:len(v) for k,v in groups.items()}},indent=2))


if __name__=='__main__':main()

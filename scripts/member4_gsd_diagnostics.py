"""Independent retrieval checks and nearest-pair diagnostics, never auto-exclusions."""
import io
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.member4_gsd_prepare import ROOT, DATA, read, cross_candidates, pair_id
from transparent_seg.near_duplicates import METHOD, image_features
from transparent_seg.prepare import write_json


def main():
    if (ROOT/'retrieval_diagnostics.json').exists():
        raise FileExistsError('Preserve previous retrieval diagnostics')
    g,t=read(ROOT/'gsd_inventory.json'),read(ROOT/'trans10k_inventory.json')
    gf=np.load(ROOT/'gsd_features.npy',allow_pickle=False)
    tf=np.load(ROOT/'trans10k_features.npy',allow_pickle=False)
    lookup=np.array([int(i).bit_count() for i in range(256)],np.uint8)
    def distance(a,b):
        xor=np.ascontiguousarray(a[:,None]^b[None,:])
        return lookup[xor.view(np.uint8).reshape(len(a),len(b),8)].sum(2,dtype=np.uint16)
    found=set(); nearest={}
    for lv,rv,limit,method in ((0,0,8,'whole_vs_whole'),(0,1,6,'whole_vs_center80'),(1,0,6,'center80_vs_whole')):
        for start in range(0,len(g),64):
            stop=min(start+64,len(g))
            pd=distance(gf[start:stop,lv,0],tf[:,rv,0]); dd=distance(gf[start:stop,lv,1],tf[:,rv,1])
            ii,jj=np.nonzero((pd<=limit)&(dd<=12))
            for i,j in zip(ii.tolist(),jj.tolist()):
                if g[start+i]['pixel_sha256']!=t[j]['pixel_sha256']:
                    found.add((g[start+i]['id'],t[j]['id']))
            for local in range(stop-start):
                # Lexicographic combined distance, pHash, dHash, then manifest order.
                rank=(pd[local].astype(np.int64)+dd[local])*10000+pd[local]*100+dd[local]
                j=int(np.argmin(rank)); i=start+local
                key=(int(rank[j]),j,method)
                if i not in nearest or key<nearest[i][0]:
                    nearest[i]=(key,{'gsd_id':g[i]['id'],'trans10k_id':t[j]['id'],
                        'trans10k_split':t[j]['split'],'pair_id':pair_id(g[i]['id'],t[j]['id']),
                        'kind':'diagnostic_not_threshold_candidate','method':method,
                        'phash_distance':int(pd[local,j]),'dhash_distance':int(dd[local,j]),
                        'phash_threshold':limit,'dhash_threshold':12,
                        'automatic_exclusion':False})
    recorded=read(ROOT/'cross_candidates.json')['near']
    assert found=={(r['gsd_id'],r['trans10k_id']) for r in recorded}
    diagnostics=[r for _,r in sorted(nearest.values(),key=lambda v:v[0])
                 if (r['gsd_id'],r['trans10k_id']) not in found][:12]
    # Known positives are created in memory only and never added to either dataset.
    controls=[]
    for i in (0,128,512):
        with Image.open(DATA/g[i]['image']) as im:
            im=im.convert('RGB'); transformed=im.resize((im.width*2,im.height*2),Image.Resampling.BICUBIC)
        buffer=io.BytesIO(); transformed.save(buffer,format='JPEG',quality=90)
        with Image.open(io.BytesIO(buffer.getvalue())) as im: feature=image_features(im)
        fake={**g[i],'id':'known_positive_control/'+str(i),'split':'control','pixel_sha256':'different_dimensions'}
        matches=cross_candidates([gf[i]],[feature],[g[i]],[fake])
        assert len(matches)==1
        controls.append({'gsd_id':g[i]['id'],'transformation':'2x resize and JPEG quality90 in memory',
            'retrieved':True,'distance':matches[0]['matched_views'],'not_an_actual_Trans10K_sample':True})
    write_json(ROOT/'retrieval_diagnostics.json',{'independent_byte_popcount_matches_vectorized_retrieval':True,
        'gsd_rows':len(g),'trans10k_rows':len(t),'cross_pair_count_per_view':len(g)*len(t),'view_count':3,
        'actual_near_candidates':len(found),'known_positive_controls':controls,
        'nearest_pair_selection':'12 distinct GSD rows with smallest combined pHash+dHash over the same three views; diagnostic only, not threshold candidates',
        'diagnostic_pairs':diagnostics,'no_overlap_claim':False})
    review=read(ROOT/'human_review_template.json')
    review['cross_audit_review']={role:{'reviewer':'','status':'pending','reason':''} for role in ('member4','member1')}
    write_json(ROOT/'human_review_template.json',review)
    print(f'Independent exhaustive retrieval agrees: {len(found)} actual near candidates; 3 known positives retrieved; {len(diagnostics)} nearest-pair diagnostics.',flush=True)


if __name__=='__main__': main()

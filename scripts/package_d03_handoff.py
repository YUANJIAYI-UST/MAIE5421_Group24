"""Package the verified D03 candidate and evidence, excluding raw datasets."""
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from transparent_seg.prepare import write_json, sha256_file


def main():
    root=Path.cwd();run=Path('runs/trans10k_d03_20261028');manifests=Path('manifests/trans10k_d03_20261028')
    read=lambda p:json.loads(Path(p).read_text())
    verification=read(run/'verification.json');audit=read(manifests/'audit.json');selection=read(manifests/'selection.json')
    assert verification['status']=='passed' and all(verification['byte_identical_reproduction'].values())
    suite=ET.parse(run/'pytest.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==52 and all(int(suite.attrib[k])==0 for k in ('failures','errors','skipped'))
    for name,digest in audit['manifest_sha256'].items():assert sha256_file(manifests/f'{name}.json')==digest
    # Recheck guards added after the long independent scan without repeating all I/O.
    base={s:read(Path('manifests/trans10k_v1_review_20261028')/f'{s}.json') for s in ('train','validation','test')}
    ti=read(run/'test/inventory.json');tv=read(run/'train_val_aligned/inventory.json')
    assert base['test']['complete_official_split'] and len(base['test']['samples'])==base['test']['official_count']==4428
    assert {s['id'] for s in base['test']['samples']}=={s['id'] for s in ti['samples']}
    ps={s:{x['pixel_sha256'] for x in base[s]['samples']} for s in base}
    assert not ps['train']&ps['validation'] and not ps['train']&ps['test'] and not ps['validation']&ps['test']
    rules=read('configs/data/trans10k_d03_rules_20261028.json')
    orientation=read('configs/data/trans10k_v1_review_20261028.json')
    assert sha256_file('runs/trans10k_data_audit_20261022/diagnostics/diagnostic_inventory.json')==orientation['source']['inventory_sha256']
    assert sha256_file('configs/data/trans10k_v1_review_20261028.json')==rules['orientation_policy_sha256']
    write_json(run/'final_guard_checks.json',{'complete_official_test':True,'exact_cross_split_overlap':0,'pinned_D01_source_inventory':True,'pinned_orientation_policy':True})
    write_json(run/'source_code_hashes.json',{p:sha256_file(p) for p in (
        'scripts/audit_trans10k_v2.py','transparent_seg/data_v2.py','transparent_seg/prepare_holdout.py',
        'scripts/verify_holdout_release.py','tests/test_holdout_data.py','configs/data/trans10k_d03_rules_20261028.json')})
    env={'python':sys.version,'platform':platform.platform(),
         'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','Pillow','scipy','pytest','pyarrow')},
         'git_base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'uncommitted_local_work':True,'gpu_pilot':'not_run','formal_training':'not_run'}
    write_json(run/'environment.json',env)
    (run/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (run/'tracked_changes.patch').write_text(subprocess.check_output(['git','diff'],text=True))
    source={'author_repository':'https://github.com/xieenze/Trans2Seg',
            'author_revision':'3972916bba7f985ca1aabc047fea56bdec9e9e5d',
            'author_google_drive_result':'404; no official archive obtained',
            'author_baidu_public_archive':{'name':'trans12cls.zip','bytes':1172033952,'downloaded':False},
            'mirror_repository':'https://huggingface.co/datasets/rdyzakya/Trans10K-v2',
            'mirror_revision':rules['mirror_revision'],'mirror_equivalence_to_author_archive':'unverified_pending_review',
            'mirror_card_discloses_one_red_to_cup_mapping':True,
            'local_category_relabeling':False,
            'parquet_sha256':{**tv['parquet_sha256'],**ti['parquet_sha256']},
            'alignment_images':10428,'unresolved_alignment_or_encoding_issues':0,
            'selection_sha256':sha256_file(manifests/'selection.json')}
    write_json('reports/TRANS10K_V2_D03_SOURCE_20261028.json',source)
    counts=read(manifests/'class_counts.json')
    before=read(run/'selection/class_counts_train_validation.json')['eligible_base_counts']
    lines=['D03 类别图片计数（同图可含多个类别，列和不等于图片总数）',
           'ID | 作者代码类别名 | 隔离前 train/val | 隔离后 train/val | test_heldout 中图片数']
    for c,name in enumerate(tv['class_names']):
        lines.append(f"{c:2d} | {name:14s} | {before['train'][c]['images']:4d}/{before['validation'][c]['images']:3d} | {counts['train'][c]['images']:4d}/{counts['validation'][c]['images']:3d} | {counts['test_heldout'][c]['images']}")
    lines+=['','逐类别原生 V2 像素数见 manifests/trans10k_d03_20261028/class_counts.json。',
             'ID 8 代码名 Floor Glass，论文名 glass wall；ID 9 代码名 Glass Bow，论文名 glass bowl。',
             '测试计数仅在选择文件冻结后生成，没有输入选类函数。']
    Path('reports/D03_CLASS_COUNTS_20261028.txt').write_text('\n'.join(lines)+'\n')
    report='''D03 已完成的技术工作与人工确认入口
日期：2026-10-28

结论
已经完成 D03 的真实数据对齐、选类统计、隔离清单、哈希、生成代码和泄漏检查。
当前是“待人工确认的镜像候选版”，不能称为全组已批准、成员 5 已验收的正式协议。

实际结果
- V2 5000/1000/4428 共 10428 张与 V1 按原 ID 和 split 一一对应。
- 18 张沿用 D01 旋转记录后，全部图像内容对齐检查通过；未知颜色 0。
- 只依据训练／验证统计，固定 ID 5 Glass Door 和 ID 8 Floor Glass。
  ID 8 在论文称 glass wall，需统一报告名称；不要理解成我们自行创造的新类别。
- 在 D01 的 4990/999 基础上排除 1729 张训练、350 张验证，剩余 3261/649。
  两类在剩余训练／验证中的图片数、像素数均为 0；混合类别图片整张移除。
- 官方测试仍完整保留 4428 张；派生 held-out 子集 1493 张、seen 子集 2935 张。
- V1 原图、二值标签路径和哈希保持不变；V2 只用来过滤和提供类别信息。
- 52 项测试通过；另直接重读实际标签并检查源文件哈希，泄漏检查通过。
- 11 个交付 JSON 在本机第二次生成后逐字节相同；尚未实际验证其他人的电脑。
- A/D pilot 配置已固定清单哈希；未运行 GPU 或正式训练。

现在需要你确认的两件事
1. 数据来源：作者 Google Drive 已失效。本次用固定版本的 Hugging Face 镜像，
   它的文件指纹和 V1 图像对应关系已核验，但未取得作者百度原 ZIP 做逐标签比对。
   镜像说明曾将一处红色映射为 cup。你可确认接受该镜像候选版本，或提供作者原包
   供进一步比对；这项差异不能省略、也不能冒称作者原包审计已完成。
2. 选类方案：确认采用 ID 5 和 ID 8，统一 ID 8 的名称。
   先看 human_review/class_05_contact_01.jpg 与 class_08_contact_01.jpg 两张图。
   当前没有替你填写任何“人工已审核”。

已有的人工任务仍然保留
D01：18 张方向图、原抽样叠图等人工复核；本次 V2 对齐图提供额外证据。
D02：153 对近重复候选的人工判断和理由。D03 完成不代表这些已经完成。
如人工复核改变 D01/D02 基础清单，必须新建版本、重建 D03 并更新哈希。
完成你的人工确认后，再把完整成果交给成员 5 做独立复核。

最重要的文件
docs/11_MEMBER1_D03_HOLDOUT.txt：完整说明、来源与复现命令。
reports/D03_CLASS_COUNTS_20261028.txt：类别计数表。
manifests/trans10k_d03_20261028/audit.json：结果、排除理由和每份清单哈希。
runs/trans10k_d03_20261028/verification.json：实际数据验证与逐字节复现结果。
runs/trans10k_d03_20261028/human_review/review_template.json：尚未填写的人工确认记录。

来源
作者仓库：https://github.com/xieenze/Trans2Seg
镜像：https://huggingface.co/datasets/rdyzakya/Trans10K-v2
镜像 revision：3445e93f1e507ea6c2dd36f06badfd5e818b699f
作者百度：https://pan.baidu.com/s/1P-2l-Q2brbnwRd2kXi--Dg  提取码 oqms

本次没有发送组员消息、推送 GitHub 或提交正式实验结果。
'''
    Path('reports/MEMBER1_D03_DELIVERY_20261028.txt').write_text(report)
    files=set()
    def include(path):
        p=Path(path)
        if p.is_dir():files.update(x for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.name!='.DS_Store')
        elif p.is_file():files.add(p)
        else:raise FileNotFoundError(p)
    for folder in ('transparent_seg','tests','scripts','configs',str(manifests),
                   'manifests/trans10k_v1_review_20261028','data/Trans10K_v1_review_20261028'):
        include(folder)
    for folder in ('selection','train_val_aligned','test','human_review'):
        include(run/folder)
    for filename in ('verification.json','verification.log','pytest.xml','pytest.log','selection_freeze_order.json',
                     'ai_visual_review.json','source_code_hashes.json','environment.json','pip_freeze.txt','tracked_changes.patch','final_guard_checks.json'):
        include(run/filename)
    for p in ('requirements.txt','requirements-data.txt','pyproject.toml','README.txt','.gitignore',
              'docs/09_TRANS10K_DATA_WORKFLOW.txt','docs/10_MEMBER1_LOCAL_RELEASE.txt','docs/11_MEMBER1_D03_HOLDOUT.txt',
              'reports/MEMBER1_D03_DELIVERY_20261028.txt','reports/D03_CLASS_COUNTS_20261028.txt','reports/TRANS10K_V2_D03_SOURCE_20261028.json',
              'reports/TRANS10K_V1_SOURCE_20261022.json','runs/trans10k_data_audit_20261022/diagnostics/diagnostic_inventory.json',
              'runs/trans10k_data_audit_20261022/diagnostics/audit.json',
              'data/archives/Trans10K_v2_mirror/README.md','data/archives/Trans10K_v2_mirror/files.json',
              'data/archives/Trans10K_v2_mirror/download_train_validation.json','data/archives/Trans10K_v2_mirror/download_test.json',
              'data/archives/Trans10K_v2/author_revision.json','data/archives/Trans10K_v2/author_code/README.md',
              'data/archives/Trans10K_v2/author_code/LICENSE',
              'data/archives/Trans10K_v2/author_code/segmentron/data/dataloader/transparent11.py',
              'data/archives/Trans10K_v2/author_code/segmentron/utils/visualize.py'):
        include(p)
    start='''D03 技术交付候选包；人工确认与成员 5 复核尚未完成。
先阅读 reports/MEMBER1_D03_DELIVERY_20261028.txt。
本包提供 D03 和所需 V1 基础元数据／18 张修复标签，不包含原始 V1/V2 图片、parquet 或权重。
依 docs/11_MEMBER1_D03_HOLDOUT.txt 下载同一来源并复现。
D01/D02 的完整人工复核图片仍在之前 Member1_V1_D01_D02_Handoff_20261028.zip 中；两份包应合并审阅。
旧包中的 D03 未完成说明是历史状态；本包为后续补充。当前 D01/D02 人工部分仍未完成。
不能把 review_template 中 pending 或 AI 记录当成人工审核结论。
'''
    archive=Path('reports/Member1_D03_Handoff_20261028.zip')
    if archive.exists():raise FileExistsError('Preserve existing handoff archive')
    checks={p.as_posix():sha256_file(p) for p in sorted(files)}
    checks['START_HERE.txt']=hashlib.sha256(start.encode()).hexdigest()
    with ZipFile(archive,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,p.as_posix())
        z.writestr('START_HERE.txt',start)
        z.writestr('CHECKSUMS.json',json.dumps(checks,indent=2)+'\n')
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(checks)|{'CHECKSUMS.json'}
        for name,digest in checks.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    digest=sha256_file(archive)
    archive.with_suffix('.sha256.txt').write_text(f'{digest}  {archive.name}\n')
    write_json(run/'handoff_verification.json',{'archive':archive.as_posix(),'sha256':digest,
               'bytes':archive.stat().st_size,'payload_files':len(checks),'zip_crc':'passed','all_payload_hashes':'passed',
               'human_review':'pending','source_mirror_acceptance':'pending'})
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest,'payload_files':len(checks)},indent=2))


if __name__=='__main__':main()

"""Prepare the final member 4 writing and presentation data after verification."""
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.summarize_member4_completion import REPORT, ROOT, SHAS, read, save, require

OUT = Path('deliverables/member4_complete_20261118')
REPO = 'https://github.com/YUANJIAYI-UST/MAIE5421_Group24'


def text(name, value):
    (OUT/name).write_text(value.strip()+'\n', encoding='utf-8')


def main():
    summary, verified = read(REPORT/'summary.json'), read(REPORT/'verification.json')
    require(verified['passed'] and summary['stage6_experiments_complete'], 'Evaluations incomplete')
    require(all(r['all_prediction_hashes_match_formal_run'] for r in read(REPORT/'gallery/index.json')),
            'Gallery not verified')
    require(not OUT.exists(), 'Preserve completed packages; choose a new version')
    OUT.mkdir(parents=True)
    evidence = OUT/'evidence'
    evidence.mkdir()
    for source in sorted(REPORT.iterdir()):
        if source.suffix in ['.json','.csv']:
            shutil.copyfile(source, evidence/source.name)
    shutil.copytree(REPORT/'figures', OUT/'figures')
    shutil.copytree(REPORT/'gallery', OUT/'gallery')
    shutil.copyfile('scripts/plot_member4_completion.py', OUT/'plot_member4_completion.py')
    shutil.copyfile('deliverables/member4_stage7_partial_20261116/member4_references.bib', OUT/'member4_references.bib')
    # Preserve earlier selection evidence supporting the already-published C claim.
    shutil.copyfile('reports/member4_stage4_20261115/failure_selection.csv', evidence/'AC_failure_selection.csv')
    r = summary['ABCD']
    g = summary['GSD']
    clean = lambda m,k='foreground_iou': r[m][k]['clean_percent']
    mean = lambda m,k='foreground_iou': r[m][k]['perturbed_mean_percent']
    drop = lambda m,k='foreground_iou': r[m][k]['drop_pp']
    change_ca, change_db = mean('C')-mean('A'), mean('D')-mean('B')
    interaction = change_db-change_ca
    clean_table = '\n'.join(f"{m:5} {clean(m):9.4f} {clean(m,'region_f1'):10.4f} "
                            f"{clean(m,'boundary_iou'):12.4f} {mean(m):13.4f} {drop(m):9.4f}" for m in 'ABCD')
    gsd_table = '\n'.join(f"{m:5} {g[m]['foreground_iou']*100:9.4f} {g[m]['region_f1']*100:10.4f} "
                          f"{g[m]['boundary_iou']*100:12.4f} {g[m]['background_fpr']*100:12.4f}" for m in 'AD')
    with (REPORT/'ABCD_condition_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    idx = {(x['model'],x['condition'],x['subset']):x for x in rows}
    conditions = sorted({x['condition'] for x in rows if x['condition'] != 'clean'})
    wins = {a+b:sum(float(idx[a,c,'all']['foreground_iou']) > float(idx[b,c,'all']['foreground_iou'])
                   for c in conditions) for a,b in [('C','A'),('D','B')]}
    report = f'''Photometric robustness and transfer to glass scenes
Member 4 contribution for the Group 24 report
Final evaluation snapshot: {datetime.now().astimezone().isoformat(timespec='seconds')}

Integration: use this text in the team's Methods, Experiments and Discussion sections.
It is a contribution to the team report, not a replacement for the complete report.
The formal main-experiment B/D models are now available and evaluated. The earlier
B02 grid and the separate E03 category-isolation checkpoints are not used here.

Photometric augmentation and controls
We test whether modest appearance changes improve transparent-object segmentation
under photometric shifts. C uses the ImageNet-initialized MiT-B0 encoder and
SegFormer-style decoder of A [1], with probability 0.5 of applying one uniformly
selected brightness, contrast or gamma operation. Its parameter is sampled from
[0.8, 1.2]. Augmentation follows resizing and precedes normalization; contrast uses
the per-channel image mean. Targets and valid-pixel masks remain unchanged. Checks
on 20 fixed training images verified label invariance, deterministic sampling,
cache equivalence and augmentation-free validation. These appearance changes do
not physically simulate refraction. B adds boundary supervision, while D combines
boundary supervision with exactly the same augmentation as C (radius 2, weight 0.1).

The reviewed Trans10K-v1 split [2] contains 4,959/995/4,428 training/validation/test
images. All four main runs share seed 42, common initial parameter hashes,
512-by-512 inputs, batch size 8, AdamW learning rate 6e-5 and 30 epochs with
18,600 updates. C starts from the common initialization, not trained A weights.
Checkpoints are selected solely by foreground IoU on all 995 validation images;
the delivered B, C and D checkpoints come from epochs 29, 30 and 28. Training uses
BF16. Evaluation uses FP32, logits interpolated to original resolution and a fixed
0.5 sigmoid threshold. IoU/F1 pool pixel counts; Boundary IoU [3] averages images.
The optional OpenCV boundary implementation was checked against the original
evaluator: B/D predictions and metrics matched on fixed validation samples under
all 13 conditions; A's 813-image GSD recheck matched every saved prediction hash
and metric. All completed chunks and image memberships were independently audited.

Full-test photometric robustness
Every model is evaluated on all 4,428 images for clean inputs and 12 independent
perturbations: brightness {{0.6,0.8,1.2,1.4}}, contrast {{0.5,0.75,1.25,1.5}} and
gamma {{0.6,0.8,1.2,1.6}}. Each condition starts from the original image. The
perturbed mean equally weights these 12 aggregate scores; clean is excluded.
The drop is clean minus this mean. Differences below are percentage points (pp).

Table 1. Full-test scores (%), foreground-IoU perturbation mean (%) and drop (pp).
Model  Clean IoU   Clean F1   Boundary IoU   Perturb mean   IoU drop
{clean_table}

C minus A gives a clean IoU difference of {clean('C')-clean('A'):+.4f} pp and a
mean-perturbed IoU difference of {change_ca:+.4f} pp. C exceeds A in {wins['CA']}/12
conditions. D minus B gives {clean('D')-clean('B'):+.4f} pp on clean inputs and
{change_db:+.4f} pp for the perturbation mean, exceeding B in {wins['DB']}/12
conditions. The descriptive difference between these augmentation effects,
(D-B)-(C-A), is {interaction:+.4f} pp. This single-seed difference is not an
interaction significance test. Easy/hard tables and all metric-specific effects
are provided in evidence/ABCD_effects_percentage_points.csv; no subgroup replaces
the full-test conclusion. Clean Boundary IoU changes by
{clean('C','boundary_iou')-clean('A','boundary_iou'):+.4f} pp for C-A and
{clean('D','boundary_iou')-clean('B','boundary_iou'):+.4f} pp for D-B.

Mean improvements do not imply that every image improves. The previously declared
gamma-1.6 regression case test/hard/images/765.jpg has A IoU 69.06% and C IoU 6.43%
(not 64.30%). Additional B/D failures are chosen by the three lowest per-image
D-minus-B IoU differences at gamma 1.6. Their selection rules and exact scores
are recorded in evidence/failure_selection.csv. These cases diagnose possible
errors; they do not estimate their frequency. No difficult images were removed.

Transfer to GSD glass scenes
We use the previously reviewed and frozen 813 test records from GSD [4], mirror
revision f8897574bbc2632a440efa1de021fd2b02f89f51. Original image and mask bytes
are preserved; 255 denotes glass. Fixed full-image and center-crop searches found
no cross-dataset duplicate candidates at the recorded thresholds. A 20-image
label review preceded scoring. Limited search cannot prove zero overlap. The
cohort retains 812 distinct decoded photographs, including one repeated photograph
with different masks. Full and supplementary manifests have identical ordered
members and share predictions; they are not independent experiments.

A and D are evaluated without GSD training, checkpoint selection or threshold
tuning. These are main-experiment checkpoints, not E03 category-isolation models.

Table 2. GSD scores (%); background FPR is lower-is-better. 813 records per model.
Model   FG IoU    Region F1   Boundary IoU   Background FPR
{gsd_table}

D-A changes foreground IoU by {summary['GSD_D_minus_A_pp']['foreground_iou']:+.4f} pp
and background FPR by {summary['GSD_D_minus_A_pp']['background_fpr']:+.4f} pp.
Thus fewer background false positives coexist with slightly lower foreground
overlap. D does not improve aggregate GSD segmentation IoU in this experiment.
Foreground recall, derived from the same pooled TP/FN counts, changes from
{g['A']['tp']/(g['A']['tp']+g['A']['fn'])*100:.4f}% to
{g['D']['tp']/(g['D']['tp']+g['D']['fn'])*100:.4f}%: D also misses more true glass
pixels. This A/D comparison changes both boundary supervision and augmentation,
so it does not isolate either component's contribution to scene transfer.
The open-window failure, gsd/test/122355684_81bf2b3d28, remains available for both
models, with ground truth and exact prediction masks. GSD tests scene transfer;
it cannot replace the separate experiment on held-out object categories.
GSD emphasizes glass surfaces, whereas the Trans10K foreground includes a broader
set of transparent objects. Scene and foreground-definition differences limit
direct comparisons between scores on the two datasets.

Limitations and contribution
All main comparisons use one seed. Training hardware and Python environments
differ (C: RTX 5090 Laptop; delivered B/D: RTX 5070 Ti), although common settings
and initialization are verified. No statistical significance or universal gain
is claimed. The current complete results supersede the earlier pending-B/D draft.
Member 4 verified augmentation, trained C, completed all four models' full
photometric comparisons, audited/froze GSD, completed A/D scene-transfer evaluation
and packaged traceable results. B/D training belongs to the delivering members.

References (merge keys with the team's bibliography)
[1] Xie et al. SegFormer. NeurIPS 2021.
[2] Xie et al. Segmenting Transparent Objects in the Wild. ECCV 2020.
[3] Cheng et al. Boundary IoU. CVPR 2021.
[4] Lin, He and Lau. Rich Context Aggregation with Reflection Prior for Glass
Surface Detection. CVPR 2021.
'''
    text('MEMBER4_Report_Section_EN.txt', report)
    tex = r'''% Member 4 contribution fragment; merge into the team's CVPR document.
% Requires booktabs and the supplied bibliography keys. Not a standalone paper.
% Check the final six-page limit after integration; no team-template compilation claimed.
\subsection{Photometric robustness and scene transfer}
\paragraph{Augmentation and controls.}
C adds photometric augmentation to the SegFormer-style baseline~\cite{xie2021segformer}.
With probability 0.5, one uniformly selected brightness, contrast or gamma operation
uses a parameter in $[0.8,1.2]$. It follows resizing and precedes normalization;
target and valid-pixel masks are unchanged. B adds boundary supervision; D adds
the same augmentation to B. All four main runs share initialization, seed 42,
$512\times512$ inputs, batch 8, AdamW learning rate $6\times10^{-5}$ and 30 epochs
(18,600 updates). The reviewed Trans10K split~\cite{xie2020transparent} has
4,959/995/4,428 training/validation/test images. Checkpoint selection uses only
validation IoU. Inference is FP32 at original image resolution with threshold 0.5.
IoU/F1 pool pixels; Boundary IoU~\cite{cheng2021boundary} averages images.

\paragraph{Photometric evaluation.}
Every model covers all 4,428 test images under 12 independent conditions:
brightness $\{0.6,0.8,1.2,1.4\}$, contrast $\{0.5,0.75,1.25,1.5\}$ and gamma
$\{0.6,0.8,1.2,1.6\}$. Each starts from the original input. Table~\ref{tab:m4-final-robust}
equally averages the 12 aggregate IoUs, excluding clean inputs. Drop is clean minus
this mean; differences use percentage points (pp).
\begin{table}[t]
\centering\small\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrr}
\toprule
Model & Clean IoU & Clean BIoU & Mean IoU & Drop\\
\midrule
@@ROBUST_ROWS@@
\bottomrule
\end{tabular}
\caption{Full-test percentages; IoU drop in pp. One training seed.}
\label{tab:m4-final-robust}
\end{table}
C--A changes mean perturbed IoU by @@CA@@ pp; D--B changes it by @@DB@@ pp.
C exceeds A in @@CA_WINS@@/12 conditions and D exceeds B in @@DB_WINS@@/12.
The descriptive difference $(D-B)-(C-A)$ is @@INTERACTION@@ pp, not a significance
test. Average effects do not imply uniform improvements: the declared gamma-1.6
failure case 765.jpg has A IoU 69.06\% and C IoU 6.43\%. Additional failures follow
the predeclared ranking in the released evidence; no test images were removed.

\paragraph{GSD scene transfer.}
The reviewed GSD mirror~\cite{lin2021glass} contributes 813 fixed records, retaining
812 unique decoded photographs and one repeated photograph with different masks.
Limited duplicate searches found no cross-dataset candidates at the recorded
thresholds; this does not prove zero overlap. Full and supplementary manifests
have identical members and share predictions. A/D use their main-experiment
checkpoints without GSD training, model selection or threshold tuning.
\begin{table}[t]
\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\toprule
Model & IoU & F1 & BIoU & BG FPR\\
\midrule
@@GSD_ROWS@@
\bottomrule
\end{tabular}
\caption{GSD percentages, 813 records per model; BG FPR is lower-is-better.}
\label{tab:m4-final-gsd}
\end{table}
D--A changes IoU by @@GSD_DELTA@@ pp and reduces background false positives,
while missing more glass pixels. It does not improve overall GSD IoU. This
comparison changes both boundary supervision and augmentation; it does not
isolate their transfer effects. GSD scene transfer is distinct from held-out
category evaluation. All results use one seed; training hardware/Python versions
differ. We retain regressions and make no significance or universal-gain claim.
GSD glass surfaces and the broader Trans10K transparent foreground also differ
in coverage, limiting direct score comparisons across datasets.
'''
    values = {
      'ROBUST_ROWS':'\n'.join(f"{m} & {clean(m):.3f} & {clean(m,'boundary_iou'):.3f} & {mean(m):.3f} & {drop(m):.3f} \\\\" for m in 'ABCD'),
      'GSD_ROWS':'\n'.join(f"{m} & {g[m]['foreground_iou']*100:.3f} & {g[m]['region_f1']*100:.3f} & {g[m]['boundary_iou']*100:.3f} & {g[m]['background_fpr']*100:.3f} \\\\" for m in 'AD'),
      'CA':f'{change_ca:+.4f}','DB':f'{change_db:+.4f}','CA_WINS':str(wins['CA']),
      'DB_WINS':str(wins['DB']),'INTERACTION':f'{interaction:+.4f}',
      'GSD_DELTA':f"{summary['GSD_D_minus_A_pp']['foreground_iou']:+.4f}",
    }
    for key,value in values.items():tex=tex.replace('@@'+key+'@@',value)
    require('@@' not in tex,'Unfilled report template')
    text('MEMBER4_Report_Section_EN.tex',tex)
    sign_db = 'increases' if change_db >= 0 else 'decreases'
    notes = [
      {'title':'Photometric augmentation','seconds':40,
       'zh':'训练时，一半照片保持原样，另一半随机改变亮度、对比度或gamma中的一种，强度在0.8到1.2。增强放在缩放后、归一化前，玻璃位置和标签完全不变。C在A基础上只加增强；D在B基础上加同样增强。四组共用初始化、数据和训练预算，所以可以分别比较C和A、D和B。',
       'en':'My part tests whether appearance changes make segmentation more stable. During training, each image has a fifty percent chance of receiving one operation: brightness, contrast, or gamma. The strength is between zero point eight and one point two. Object positions and masks stay unchanged. C adds augmentation to A. D adds the same augmentation to B. The four models share their initialization, data, and training budget.',
       'source':'configs/member4_full_C_20261115.json; evidence/intake_verification.json; stage-2 verified augmentation example'},
      {'title':'Four-model clean test','seconds':40,
       'zh':'这一页是正常照片的考试，四组均测试4428张。IoU看预测区域与答案的重合程度，Boundary IoU看轮廓。数值越高越好。四组分数接近，表格的小数差值不能夸大成显著提高。模型只用验证集选，不能根据这里的测试分挑模型。',
       'en':f'All four models are tested on the same four thousand four hundred and twenty-eight clean images. IoU measures how well the predicted region overlaps the ground truth. Boundary IoU focuses on the outline. Higher is better. C reaches {clean("C"):.2f} percent IoU, and D reaches {clean("D"):.2f} percent. The differences are small. We selected checkpoints using validation data only, and we do not claim a significant improvement.',
       'source':'evidence/ABCD_clean_results.csv; evidence/intake_verification.json'},
      {'title':'Photometric robustness','seconds':55,
       'zh':f'三张图依次看亮度、对比度、gamma。横轴是变化强度，纵轴是IoU，1表示不变。每个点都有完整4428张图，均值只包括12个扰动点。纵轴放大了，所以曲线距离不代表巨大收益。C比A平均提高{change_ca:.3f}个百分点；D减B为{change_db:+.3f}个百分点。强调单seed的平均结果，个别图仍可能明显变差。',
       'en':f'The three plots show brightness, contrast, and gamma changes. The horizontal axis shows strength, with one meaning unchanged input; the vertical axis shows IoU. Every point uses the full test set. The axis is zoomed, so small gaps should not be exaggerated. Across twelve perturbations, C gains {abs(change_ca):.3f} percentage points over A. Adding augmentation to B {sign_db} the mean by {abs(change_db):.3f} percentage points. Some individual images still become worse.',
       'source':'evidence/ABCD_condition_results.csv; evidence/ABCD_robustness_summary.csv; figures/ABCD_robustness_all.png'},
      {'title':'Transfer to GSD','seconds':45,
       'zh':'最后换到813条GSD玻璃场景，不再训练、不调整阈值。A的IoU为68.13%，D为67.97%，说明D没有提升整体IoU。D背景误报更少，但这不等于所有指标都更好。失败图用固定的低分规则选择，不能代表错误出现频率。GSD是在新场景测试，不等同于新类别测试。',
       'en':'Finally, we test A and D on eight hundred and thirteen GSD records, without training or tuning on GSD. A reaches sixty-eight point one three percent IoU; D reaches sixty-seven point nine seven percent. D produces fewer background errors, but also misses more glass pixels. Overall IoU does not improve. The example shows a failure around an open window. This evaluates new scenes, separate from the experiment on unseen object categories.',
       'source':'evidence/GSD_AD_transfer_results.csv; evidence/failure_selection.csv; gallery/index.json'},
    ]
    save(OUT/'presentation_data.json', {'summary':summary,'slides':notes,'wins':wins,
         'augmentation_effects_pp':{'C_minus_A':change_ca,'D_minus_B':change_db,'difference_of_effects':interaction}})
    count = sum(len(n['en'].split()) for n in notes)
    text('MEMBER4_Speaker_Notes_ZH_EN.txt', f'''成员4四页展示讲稿：建议约3分钟，英文约{count}词。
对应小组旧v3的增强部分、第9页、第10页和第11页；整合时沿用小组最终页码。
第1页约40秒，第2页约40秒，第3页约55秒，第4页约45秒。先排练一次再微调停顿。
IoU：预测区域和答案重合得有多好；F1：区域精确率和召回率的综合。
Boundary IoU：边缘重合程度。三者越高越好。
Background FPR：原本是背景却被判成玻璃的比例，越低越好。
pp=percentage points=百分点，不是相对提高百分比。
亮度系数大于1整体变亮，小于1变暗；对比度小于1更灰、差异更小。
gamma按像素强度的幂变换：大于1通常变暗，小于1通常变亮；1表示不变。
这些操作只改变外观，不改变玻璃的位置，也不是真实折射仿真。

'''+ '\n\n'.join(f"第{i+1}页 {n['title']}（{n['seconds']}秒）\n中文理解：{n['zh']}\n\nEnglish:\n{n['en']}\n\n证据：{n['source']}" for i,n in enumerate(notes)))
    text('README_使用与验收.txt', f'''成员4完整交付包（运行编号20261118）
打包时间：{datetime.now().astimezone().isoformat(timespec='seconds')}

先读什么
报告负责人：MEMBER4_Report_Section_EN.txt，按全组篇幅精简整合；引用合入总参考文献。
LaTeX可用同名.tex精简片段和member4_references.bib；尚未在全组模板下编译，请合稿后检查6页限制。
展示负责人：MEMBER4_Presentation_4slides.pptx及MEMBER4_Speaker_Notes_ZH_EN.txt。
PPT可直接打开编辑，文字和结果表格可编辑；曲线附Python源文件，无需运行制作脚本才能使用PPT。
图表：figures/内PNG用于PPT，PDF/SVG用于报告；全部来自实际统计，不含虚构误差棒。
原表：evidence/，按all/easy/hard分组。gallery/为固定样本和明确标记的失败案例。
verification.json负责结果核验，validation.json负责最终交付文件核验；缺少后者表示打包未结束。
Git分支保存文稿、PPT、摘要和曲线；完整gallery与逐图数值归档通过最终Release取得。
逐文件SHA校验请在完整ZIP解压目录中进行，仓库精简目录不包含所有gallery图片。

完成范围
阶段1—5沿用已验收成果；正式C训练、A/C评估、GSD冻结没有被覆盖。
正式B/D已接入：初始化匹配，30轮、18600次更新，B最佳29轮、D最佳28轮。
阶段6已完成：四模型52组条件，每组4428张；GSD A/D每组813条。
GSD完整/补充清单成员完全相同，共用预测，不能算两次独立实验。
阶段7最终以validation.json的通过状态为准；旧partial包保留历史状态，本包优先。
另附E03_table_cross_review.json：完成原分工中的召回表复核，66行可回算，
seen/held-out成员和类别分母一致；这是记录/算术核验，没有重跑E03模型。

直接验收
1. evidence/verification.json passed=true，52组测试与A/D GSD全部通过。
2. 逐图统计、分块摘要和原始图像ID检查通过；TP/FP/FN/TN与导出表一致。
3. evidence/ABCD_clean_results.csv为4行，ABCD_condition_results.csv为156行（52组×3子集）。
4. gallery/index.json每张预测哈希与正式评估一致；失败图有选择依据。
5. PPT恰好4页，备注和独立讲稿齐全，数值与表格一致，无Pending B/D占位。
6. SHA256SUMS.txt可用Get-FileHash逐个对照；按源代码重新生成图不需要训练。

无需再请组员重复的工作
成员1不用重新查GSD或重做人工复核；成员3不用重训/重复上传已交付B/D。
成员2/3合并分支时保留C协议入口及B/D/E03入口，不能整文件覆盖protocol.py。
成员5只需复核最终表述和指标口径，并整合全组报告；这不是重跑实验要求。
PR #3仍不合并。主分支集成、全组总稿和彩排由小组安排。

链接
成员4分支：{REPO}/tree/member4/augmentation-gsd
成员4 PR：{REPO}/pull/3
C模型：{REPO}/releases/tag/member4-c-augmentation-20261116
正式B/D：{REPO}/releases/tag/group24-b03-b04-20261118
成员2来源：{REPO}/tree/224d008c6b3a2ac27403ec86a0e71b972bcb5da2

结果解读
C-A扰动均值差：{change_ca:+.4f}个百分点；D-B：{change_db:+.4f}个百分点。
GSD A/D IoU：{g['A']['foreground_iou']*100:.4f}% / {g['D']['foreground_iou']*100:.4f}%。
结果不理想不等于实验没完成；不修改参数或删除困难图片来改善已报告成绩。
单seed、不同训练环境，不作统计显著性保证；GSD迁移不等于未见类别泛化。
本包是成员4贡献，不能代表整个小组已完成所有课程要求。
''')
    text('REPRODUCE_复现说明.txt', fr'''在仓库根目录运行，使用项目.venv Python；模型/data不包含在文稿包。
GPU实测：NVIDIA GeForce RTX 5090 Laptop GPU，24GB。
推理环境与源代码哈希：evidence/protocol.json和原始运行identity.json；完整模型哈希见下方。
先按现有项目的数据导入说明恢复固定清单，禁止更换样本、阈值或分辨率。
安装可选边界计算依赖（不替换numpy）：
.\.venv\Scripts\python.exe -m pip install --no-deps --target .local_deps/member4_opencv_4_12 opencv-python-headless==4.12.0.88
$env:PYTHONPATH=(Resolve-Path .local_deps/member4_opencv_4_12).Path

独立重跑示例（用新的输出目录，不覆盖正式记录）：
.\.venv\Scripts\python.exe -m transparent_seg.evaluate_resumable --checkpoint runs/member4_BD_delivery_20261118/full_runs/B03_D/best.pt --manifest manifests/member1_final_20261029/v1/test.json --output runs/my_D_recheck --device cuda --conditions all --chunk-size 25 --boundary-backend opencv
中断后仅在代码版本、依赖、模型和清单不变时添加 --resume。
GSD命令将manifest改为manifests/member4_gsd_reviewed_v2_20261116/test_full.json，conditions改为clean，使用独立输出目录。
主实验D不得替换为E03 D。A/C和GSD原始A可直接复用已核验记录，无需重训。

在项目仓库中重新生成曲线（普通Python，matplotlib 3.9.0）：
python scripts/plot_member4_completion.py --report reports/member4_completion_20261118
在本交付目录中独立生成：
python plot_member4_completion.py --report evidence
此操作在evidence/figures另存曲线，不读取模型或运行训练。

直接取得B/D推理权重：
在正式B/D Release下载B03_B_best.pt、B03_D_best.pt，分别保存为
runs/member4_BD_delivery_20261118/full_runs/B03_B/best.pt和B03_D/best.pt。
加载器从checkpoint读取实际配置；只做推理不必下载完整训练归档。
要复查完整训练记录，再取group24_B03_seed42_full_runs.zip；其原始内部前缀为
runs/member2_b03_e03_20261118/，按前缀恢复到新目录，勿覆盖本机正式运行。

完整运行结果本机路径：
A测试：runs/member4_A_full_test_20261115
C测试：runs/member4_C_full_test_20261115
B/D测试：runs/member4_completion_20261118/B_test、D_test
A GSD：runs/member4_stage6_20261116/A_full
D GSD：runs/member4_completion_20261118/D_GSD
本轮源代码快照及日志：runs/member4_completion_20261118/source_snapshot、logs
为利用同一GPU的余量，部分D分块在B评估期间提前计算；B接近完成前受控暂停辅助D进程树，
由原队列恢复完整分块。详情见scheduling_appendix.json及D_parallel_state.json。
计算配置未改；不要把最后一次续跑的elapsed_seconds误写为整个D模型的完整评估耗时。
启动时Git HEAD为3d11d48397840443d7274d65dd03c44c7b73a0bf，未提交的已核验后端扩展以源文件哈希/快照固定。
后续提交文稿后的Git HEAD不同；旧分块拒绝跨版本续跑是预期保护，请保留原始快照。

模型SHA256
'''+ '\n'.join(f'{m} {sha}' for m,sha in SHAS.items()))
    save(OUT/'packaging_status.json', {'writing_and_evidence_complete':True,
        'pptx_validation_pending':True,'stage7_complete':False})
    print(json.dumps({'package':str(OUT),'english_words':count,'CA_pp':change_ca,'DB_pp':change_db}),flush=True)


if __name__ == '__main__':
    main()

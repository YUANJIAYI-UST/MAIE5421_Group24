"""Generate report-quality figures from the independently audited full results."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

STYLES = {
 'A': dict(color='#0072B2', marker='s', linestyle='--', label='A: baseline'),
 'B': dict(color='#009E73', marker='^', linestyle='--', label='B: boundary'),
 'C': dict(color='#D55E00', marker='o', linestyle='-', label='C: augmentation'),
 'D': dict(color='#CC79A7', marker='D', linestyle='-', label='D: combined'),
}
CONDITIONS = {'brightness': [.6, .8, 1.2, 1.4], 'contrast': [.5, .75, 1.25, 1.5], 'gamma': [.6, .8, 1.2, 1.6]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('reports/member4_completion_20261118'))
    args = parser.parse_args()
    report = args.report
    assert json.loads((report/'verification.json').read_text())['passed']
    source = report/'ABCD_condition_results.csv'
    with source.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    lookup = {(r['model'], r['condition'], r['subset']): r for r in rows}
    assert len(lookup) == len(rows) == 156
    for row in rows:
        iou = float(row['foreground_iou'])
        tp, fp, fn = (int(row[k]) for k in ('tp', 'fp', 'fn'))
        assert math.isclose(iou, tp/(tp+fp+fn), abs_tol=1e-12)
        assert int(row['images']) == {'all': 4428, 'easy': 3491, 'hard': 937}[row['subset']]
    out = report/'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.titlesize': 15, 'axes.labelsize': 13, 'xtick.labelsize': 11,
        'ytick.labelsize': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    summaries = []
    for models, group in [(['A','B','C','D'], 'all'), (['A','C'], 'all'),
                          (['B','D'], 'all'), (['A','B','C','D'], 'easy'),
                          (['A','B','C','D'], 'hard')]:
        fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.25), sharey=True)
        fig.subplots_adjust(left=.070, right=.985, bottom=.24, top=.70, wspace=.22)
        fig.suptitle('Robustness to Photometric Perturbations', y=.975, fontsize=19, fontweight='bold')
        count = {'all':4428, 'easy':3491, 'hard':937}[group]
        fig.text(.5, .90, f'Trans10K {group} test images: {count:,} per condition', ha='center', color='#444444')
        values = [float(lookup[m, f'{op}_{v:g}', group]['foreground_iou'])*100
                  for m in models for op, vs in CONDITIONS.items() for v in vs]
        # Shared, openly zoomed axes; no smoothing, error bars or hidden points.
        low, high = math.floor(min(values)), math.ceil(max(values))
        if low == high: high += 1
        for i, ((op, vs), ax) in enumerate(zip(CONDITIONS.items(), axes)):
            ax.set_title(f'({chr(97+i)}) {op.capitalize()}', pad=12)
            ax.axvline(1, color='#888888', linewidth=1, linestyle=':')
            for m in models:
                ys = [float(lookup[m, f'{op}_{v:g}', group]['foreground_iou'])*100 for v in vs]
                ax.plot(vs, ys, linewidth=1.9, markersize=6, **STYLES[m])
            ax.set_xticks(sorted(set(vs+[1])), [f'{v:g}' for v in sorted(set(vs+[1]))])
            ax.set_xlabel('Gamma exponent' if op == 'gamma' else op.capitalize()+' factor')
            ax.set_ylim(low, high)
            ax.set_yticks(list(range(low, high+1)))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.tick_params(labelleft=True)
            ax.grid(axis='y', color='#DDDDDD', linewidth=.6)
            ax.spines[['top','right']].set_visible(False)
        axes[0].set_ylabel('Foreground IoU (%)')
        handles = [Line2D([0],[0], linewidth=1.9, markersize=6, **STYLES[m]) for m in models]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.862), ncol=len(models), frameon=False)
        means = {m: math.fsum(float(lookup[m,f'{op}_{v:g}',group]['foreground_iou'])*100
                              for op, vs in CONDITIONS.items() for v in vs)/12 for m in models}
        fig.text(.5,.105,'12-condition mean:  '+'    '.join(f'{m} = {means[m]:.3f}%' for m in models),
                 ha='center', fontsize=13, fontweight='bold')
        fig.text(.5,.040,f'Single seed; zoomed y-axis ({low}–{high}%); 1.0 = identity guide; clean not plotted or averaged.',
                 ha='center', fontsize=10.5, color='#555555')
        stem = out/(''.join(models)+'_robustness_'+group)
        for extension in ['png','pdf','svg']:
            fig.savefig(stem.with_suffix('.'+extension), dpi=300)
        if models == ['A','B','C','D'] and group == 'all':
            # A separate slide figure uses larger type and leaves the slide to
            # supply its title and conclusion. The underlying points are identical.
            talk, talk_axes = plt.subplots(1,3,figsize=(15,5.1),sharey=True)
            talk.subplots_adjust(left=.072,right=.99,bottom=.18,top=.77,wspace=.22)
            for (op,vs),ax in zip(CONDITIONS.items(),talk_axes):
                ax.set_title(op.capitalize(),fontsize=21,pad=10)
                ax.axvline(1,color='#888888',linewidth=1.4,linestyle=':')
                for m in models:
                    ys=[float(lookup[m,f'{op}_{v:g}',group]['foreground_iou'])*100 for v in vs]
                    ax.plot(vs,ys,linewidth=2.2,markersize=7,**STYLES[m])
                ticks=sorted(set(vs+[1]))
                ax.set_xticks(ticks,[f'{v:g}' for v in ticks])
                ax.tick_params(labelsize=17,labelleft=True)
                ax.set_ylim(low,high);ax.set_yticks(list(range(low,high+1)))
                ax.set_xlabel('Exponent' if op=='gamma' else 'Factor',fontsize=19)
                ax.grid(axis='y',color='#DDDDDD',linewidth=.7)
                ax.spines[['top','right']].set_visible(False)
            talk_axes[0].set_ylabel('Foreground IoU (%)',fontsize=19)
            talk.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.01),
                        ncol=4,frameon=False,fontsize=18,columnspacing=1.2,handlelength=2)
            talk.savefig(out/'ABCD_robustness_presentation.png',dpi=250)
            plt.close(talk)
        plt.close(fig)
        summaries.append({'figure':stem.name,'models':models,'subset':group,'images':count,
                          'mean_percent':means,'y_axis_percent':[low,high]})
    (out/'plot_provenance.json').write_text(json.dumps({
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'matplotlib':matplotlib.__version__,'clean_plotted':False,
        'identity_guide_is_not_a_measured_point':True,'figures':summaries},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries),flush=True)


if __name__ == '__main__':
    main()

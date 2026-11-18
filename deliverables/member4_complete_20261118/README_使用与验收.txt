成员4完整交付包（运行编号20261118）
打包时间：2026-11-19T02:54:34+08:00

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
成员4分支：https://github.com/YUANJIAYI-UST/MAIE5421_Group24/tree/member4/augmentation-gsd
成员4 PR：https://github.com/YUANJIAYI-UST/MAIE5421_Group24/pull/3
C模型：https://github.com/YUANJIAYI-UST/MAIE5421_Group24/releases/tag/member4-c-augmentation-20261116
正式B/D：https://github.com/YUANJIAYI-UST/MAIE5421_Group24/releases/tag/group24-b03-b04-20261118
成员2来源：https://github.com/YUANJIAYI-UST/MAIE5421_Group24/tree/224d008c6b3a2ac27403ec86a0e71b972bcb5da2

结果解读
C-A扰动均值差：+0.1766个百分点；D-B：+0.0180个百分点。
GSD A/D IoU：68.1273% / 67.9728%。
结果不理想不等于实验没完成；不修改参数或删除困难图片来改善已报告成绩。
单seed、不同训练环境，不作统计显著性保证；GSD迁移不等于未见类别泛化。
本包是成员4贡献，不能代表整个小组已完成所有课程要求。

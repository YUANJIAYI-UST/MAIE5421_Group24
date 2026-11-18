成员4阶段7交付包：已完成结果的报告与展示材料
整理日期：2026-11-16
分支交付日期：2026-11-17（目录日期保留原结果快照日期）

一、当前范围
本包完成阶段7中不依赖B/D的成果整理，使用已冻结的A/C与GSD-A结果。
没有重新训练、调参、修改正式评分、重选测试样本或合并PR。
完整阶段6、最终四模型结论和阶段7最终版仍待正式B/D及后续评估。
历史阶段文档中的“尚未进入阶段7”“待验收”保留当时状态，不回写历史记录。
本包随member4/augmentation-gsd分支交付，并纳入现有PR #3的审阅范围。
这是可供小组整合的阶段7部分交付；B/D及最终比较仍待补充，PR保持未合并。
可在分支中直接查看本目录，或下载deliverables/member4_stage7_partial_20261116.zip整体解压。

二、先看哪些文件
1. MEMBER4_Report_Section_EN.txt
   可直接给负责report整合的组员。包含方法、实验协议、A/C表格、GSD-A结果、局限和责任说明。
   这是成员4的部分，不是全组完整报告。老师要求全组6页CVPR格式，参考文献不计页数；
   组员可按篇幅拆入Methodology、Experiments、Discussion和贡献说明。
2. MEMBER4_Report_Section_EN.tex 与 member4_references.bib
   用LaTeX/Overleaf整合时，使用这个更紧凑的正文片段和参考文献条目。
   .tex不是独立文档：将片段放入总稿，或在总稿正文使用
   \input{MEMBER4_Report_Section_EN}
   把bib条目合并进组内bibliography，已有同一文献时统一cite key，避免重复。
   表格为可编辑LaTeX表格，无需截图；本机未编译团队CVPR模板，合稿时需检查浮动位置和6页限制。
3. MEMBER4_Presentation_3slides.pptx
   恰好3页：增强方法 → A/C结果 → GSD迁移与失败案例。
   英文正文、可编辑文字和表格，真实图像已嵌入，无需本机data目录也能展示。
   每页备注含中英文讲稿与来源。可直接复制3页进小组总PPT，保留格式后复查。
4. MEMBER4_Speaker_Notes_ZH_EN.txt
   每页有中文解释、简单英文稿、展示提示和常见问题。目标总时长约2至2.5分钟。
   英文口述时不用逐项念表格数字，重点说“做了什么、结果怎样、限制是什么”。
5. evidence/ 与 figures/
   既有表格、曲线、核验摘要、失败案例和接手说明的原字节副本。
   provenance.json记录来源路径、SHA256、模型/数据版本；validation.json是本次文稿验收结果。

三、目前可以讲出的结论
正常图片：A/C IoU为87.4364%/87.5004%；C的正常边界IoU略降0.0283个百分点。
光度扰动：平均IoU从86.7838%到86.9604%，提高0.1766个百分点；hard子集提高0.3990个百分点。
换到GSD：A直接测试813条记录，IoU为68.1273%；这是A的迁移基线，尚无A/D比较。
这些是单seed结果，没有显著性检验。不能写“全面超越”“显著提高”或“边界模块已验证有效”。
GSD换场景结果不能代替类别隔离实验；完整集和补充集成员相同，不能算两次独立实验。

四、组员接手所需最少工作
报告整合者：合入英文段落和所需表格；统一术语、引用编号和全组版式，保留B/D Pending。
展示整合者：复制3页，检查字体/图片/备注，在整体约10分钟内为成员4保留约2分钟。
成员3：交付已有正式B和D运行目录或下载链接及代码commit即可，不必另写多套材料。
目录应包含best.pt、实际配置、完整训练/验证日志、最佳轮次及验证选模记录、
公共初始化哈希和环境记录。主实验B/D不能用pilot或类别隔离权重替代。
收到后由成员4核对公共设置并补B/D的全量clean+12扰动、D的GSD评估，再更新对应表和结论。
不要求其他人重做增强检查、GSD人工复核、完整训练或现有A/C评估。

五、C模型最快加载与单图验收
已有分支、依赖准备好的项目根目录执行：
  python scripts/download_member4_C.py --output runs/member4_C_release_20261116 --records
  python -m transparent_seg.predict --checkpoint runs/member4_C_release_20261116/best.pt --image runs/member4_C_release_20261116/records/demo/input.jpg --output runs/member4_stage7_handoff_demo --device auto
Windows本项目虚拟环境可将python换成.\.venv\Scripts\python.exe。
输出目录必须是新目录；已存在时换一个名字，不覆盖历史结果。
打开新目录下mask.png与overlay.png，应与示例原图同尺寸444×613。
CPU也可单图演示：将--device auto替换为--device cpu。
本次沿用已发布模型和既有下载验收，不因撰写报告重新运行正式实验。
更完整的复现步骤见evidence/28_MEMBER4_C_RELEASE_HANDOFF.txt和27_MEMBER4_STAGE5_6_BRANCH_DELIVERY.txt。
完整重跑仍需原始数据、对应环境和源码快照；不要跨代码版本复用旧评估分块。

六、链接与关键哈希
分支：https://github.com/YUANJIAYI-UST/MAIE5421_Group24/tree/member4/augmentation-gsd
C Release：https://github.com/YUANJIAYI-UST/MAIE5421_Group24/releases/tag/member4-c-augmentation-20261116
PR：https://github.com/YUANJIAYI-UST/MAIE5421_Group24/pull/3
C best.pt SHA256：69841f34ec0c76b0a2dbdce3a07f8ea5926a6beec847c4e8b03e587efc3b5f0d
资料整理基于本地分支145a2a4d69ea52fd2c330fef7122e51684d8f8bf及原运行记录。
正式训练时的代码身份以原源码哈希和快照为准，不用当前commit冒充训练时版本。

七、你的验收清单
[ ] 打开PPT，确认只有3页、照片可见，表格可编辑，备注可读。
[ ] 用中文解释“每张图一半概率，只做一种增强，标签不变”。
[ ] 能说明0.1766是百分点，平均值不包含clean，提升小且有个例退步。
[ ] 能说明GSD当前只有A；B/D、A/D比较和边界模块结论仍待补充。
[ ] 给报告整合者正文/tex/bib和所需表格；给展示整合者PPT与讲稿。
[ ] 合入全组模板后复查6页报告和约10分钟展示，不把本片段单独当成全组提交物。
未完成的最终验收：组员独立加载C并预测、B/D接入后的完整表格与结论、团队总稿版式与彩排。

MAIE5421 Topic 10 / Group 24
Boundary-Aware Transparent Object Segmentation under Appearance Variation

这是依据课程 Guidelines v3 和本组 proposal 编写的 SegFormer-B0 训练、推理与评估实现。
成员 2 已跑完冻结数据上的完整 A 基线：30 epochs / 18600 optimizer steps，seed 42。
最佳模型按全部 995 张验证图选择（epoch 29）；最终验证/测试结果见下方交接入口。
这是成员 1 清理后的派生 train/val，保留完整官方 test；不等同未过滤官方训练协议。
合成数据的 smoke test 只验证软件流程，不能作为透明物体分割效果或报告成绩证据。

先读 docs/00_START_HERE.txt，再读 docs/05_VALIDATION_GUIDE.txt。
五人分工与日期：docs/02_FIVE_MEMBER_DEVELOPMENT_PLAN.txt
成员 2 续训、资源测量与当前交接：docs/14_MEMBER2_TRAINING_HANDOFF.txt
成员 2 真实 V1 GPU pilot 与四图过拟合结果：docs/15_MEMBER2_REAL_RUNS_20261114.txt
成员 2 V2 掩码核验及 D03 A/D GPU pilot：docs/16_MEMBER2_D03_PILOT_20261114.txt
成员 2 完整 A、结果与下载说明：docs/17_MEMBER2_FULL_BASELINE_20261114.txt
机器可读结果、逐图指标、训练曲线与资源记录：reports/member2_full_A_20261114/
模型/日志：https://github.com/YUANJIAYI-UST/MAIE5421_Group24/releases/tag/member2-a-baseline-20261114
后续实验数据：https://github.com/YUANJIAYI-UST/MAIE5421_Group24/releases/tag/trans10k-course-data-20261114
成员4已完成正式C、全量A/C正常及12条件扰动对比、GSD冻结及A迁移评估；正式B/D和完整A/D比较仍待补充。
成员4报告、3页PPT和中英文讲稿（2026-11-17分支交付）：docs/29_MEMBER4_STAGE7_REPORT_PRESENTATION.txt
交付目录：deliverables/member4_stage7_partial_20261116/；同名ZIP可整体下载后直接交给报告/展示整合者。
C模型下载和单图验收：docs/28_MEMBER4_C_RELEASE_HANDOFF.txt
成员3训练/优化器独立review、三种子和held-out正式实验以各自后续交付为准，不视为已完成。
待优化任务（TXT）：docs/03_OPTIMIZATION_BACKLOG.txt
数据集与官方下载：docs/04_DATASETS_AND_DOWNLOAD.txt
老师要求逐项验收：docs/01_REQUIREMENTS_AND_ACCEPTANCE.txt
GitHub 协作：docs/06_GITHUB_COLLABORATION.txt
本次实际验证：reports/IMPLEMENTATION_STATUS.txt

成员 1 D01–D03 最终数据交付（2026-10-29）：docs/13_MEMBER1_PR_REVIEW.txt
GitHub 评审摘要与清单哈希：reports/member1_final_20261029/
完整清单、图像复核材料和已填写 Excel 通过数据 Release 内交付 ZIP 共享；不进入 Git 历史。
成员 5 分支已有独立自动复核与有限人工抽查，证据见 docs/17；冻结清单中的历史状态字段未改写。
Mac 快速验证：docs/08_MAC_QUICKSTART.txt。

Windows 快速验证（在项目根目录的 PowerShell 执行）：
  .\scripts\setup.ps1 -Device cuda
  .\.venv\Scripts\python.exe -m pytest -q
  .\scripts\smoke.ps1 -Run runs/smoke_new

目录：
  transparent_seg/  数据、模型、损失、训练、指标、评估、单图预测
  configs/          统一预算与 A/B/C/D 消融、pilot、smoke 配置
  tests/            标签、边界、指标、初始化等回归测试
  scripts/          安装、合成数据、端到端检查、顺序运行消融
  docs/             中文 TXT 计划、验收、数据来源和优化任务
  reports/          已执行的检查记录及注明协议范围的真实实验结果
  deliverables/     成员4报告、展示稿、讲稿和可追溯的配套资料
  .github/          自动检查与 PR 模板

正式数据、权重、环境、原始课程 PDF 不进入 Git 历史；由 .gitignore 排除。
用户授权的数据和模型通过 GitHub Releases 分发，附 SHA256 与还原说明。
运行命令均从仓库根目录执行。项目使用 Python 3.10+；完整 A 使用 Python 3.11.13，CI 使用 3.10。
所有论文分数必须来自真实、冻结协议的独立测试，允许负结果。

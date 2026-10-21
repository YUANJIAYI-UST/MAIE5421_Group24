MAIE5421 Topic 10 / Group 24
Boundary-Aware Transparent Object Segmentation under Appearance Variation

这是依据课程 Guidelines v3 和本组 proposal 编写的可运行初步实现。
包含真实的 SegFormer-B0 网络与训练、推理、评估流程；正式 Trans10K 实验尚未完成。
合成数据的 smoke test 只验证软件流程，不能作为透明物体分割效果或报告成绩证据。

先读 docs/00_START_HERE.txt，再读 docs/05_VALIDATION_GUIDE.txt。
五人分工与日期：docs/02_FIVE_MEMBER_DEVELOPMENT_PLAN.txt
待优化任务（TXT）：docs/03_OPTIMIZATION_BACKLOG.txt
数据集与官方下载：docs/04_DATASETS_AND_DOWNLOAD.txt
老师要求逐项验收：docs/01_REQUIREMENTS_AND_ACCEPTANCE.txt
GitHub 协作：docs/06_GITHUB_COLLABORATION.txt
本次实际验证：reports/IMPLEMENTATION_STATUS.txt

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
  reports/          已执行的检查记录；不是正式 benchmark 结果
  .github/          自动检查与 PR 模板

正式数据、权重、环境、原始课程 PDF 不上传 Git；由 .gitignore 排除。
运行命令均从仓库根目录执行。项目使用 Python 3.10+；当前测试环境为 3.10。
所有论文分数必须来自真实、冻结协议的独立测试，允许负结果。

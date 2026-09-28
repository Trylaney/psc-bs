ContextBench v2.7 PSC-BS — Fresh CPU Confirmation 一键包

Windows：
1. 解压整个 ZIP 到英文路径（例如 D:\v27）
2. 双击 RUN_ONE_CLICK.bat
3. 不需要手动下载数据；脚本会自动下载锁定的 ACS 2023 五州、Law School、UCI Diabetes 130 数据。
4. 首次会创建 .venv 并安装依赖。
5. 中途中断后重新双击即可：数据会复用，已完成 task JSON 会跳过。
6. 完成后上传 V27_FRESH_CPU_RESULTS.zip。

重要：
- v2.7 PSC-BS 方法文件有 SHA256 freeze check；不要修改 v27_psc_method.py。
- Fresh protocol 已锁定：ACS 2023 CO/MI/MN/NJ/OR + 7 ACS tasks + Law School + Diabetes Hospital；5 splits；n=32/64/128/256；held-out ExtraTrees/XGBoost/LightGBM/MLP。
- 不要在看完 fresh 结果后修改 gates 再把同一批结果称作 confirmation。

结果位置：
- runs\v2_7_fresh_cpu\TASK_MANIFEST.json
- runs\v2_7_fresh_cpu\analysis\FRESH_CPU_GATE_REPORT.md
- V27_FRESH_CPU_RESULTS.zip

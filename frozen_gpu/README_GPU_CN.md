# ContextBench v2.7 PSC-BS — Fresh GPU Confirmation

这是 **fresh CPU confirmation 已通过之后**冻结的 GPU confirmation 包。

CPU fresh：5760/5760，0 error，9/9 datasets，8/8 frozen CPU gates PASS。
GPU 阶段不会重新选择 context，只重放 CPU 阶段已经冻结的 180 个 context cells。

## 实验规模

- 9 datasets
- 5 outer splits
- context budget: 32 / 64 / 128 / 256
- 8 selectors
- TabPFN 3.5 + TabICL v2
- 正式任务：**2880**

## 第一次在 GPU 实例上

如果当前实例已经成功跑过 TabPFN3.5 / TabICL，可以跳过安装。否则：

```bash
cd /root/autodl-tmp/ContextBench_v2_7_GPU_FRESH_ONECLICK_src && ./INSTALL_GPU_ENV.sh
```

Hugging Face cache 默认复用 `/root/autodl-tmp/cache`。TabPFN 3.5 仍需要 Prior Labs license/API key；如果此前已经接受并缓存，不需要重复操作。

## 先 smoke（推荐）

```bash
cd /root/autodl-tmp/ContextBench_v2_7_GPU_FRESH_ONECLICK_src && ./RUN_GPU_SMOKE.sh
```

成功应为 12/12、0 error。

## 正式跑：自动识别 1/2/3/4... 张 GPU

后台启动：

```bash
cd /root/autodl-tmp/ContextBench_v2_7_GPU_FRESH_ONECLICK_src && ./START_GPU_FINAL_BACKGROUND.sh
```

查看进度：

```bash
cd /root/autodl-tmp/ContextBench_v2_7_GPU_FRESH_ONECLICK_src && bash CHECK_PROGRESS.sh
```

最终目标：2880 / 2880。

查看总日志：

```bash
tail -f /root/autodl-tmp/ContextBench_v2_7_GPU_FRESH_ONECLICK_src/logs/gpu_master.log
```

## 跑完

脚本自动 merge、按预冻结 GPU gates 分析并打包：

`V27_FRESH_GPU_RESULTS.zip`

把这个 ZIP 发回 ChatGPT 即可做最终 bootstrap / 分层 / 论文级审计。

## 数据

脚本会自动下载锁定的 ACS-2023 CO/MI/MN/NJ/OR、Law School 和 UCI Diabetes 130 数据，并执行 FIX3 schema adapter；计算前 `PRECHECK_V27_GPU.py` 必须通过 9/9 datasets + 180/180 contexts。

## 冻结性

- PSC-BS method SHA256: `f02ccffec4bd20225e261bd4a2b88c803ebfbf700e8f53d6b664e42f82dc307d`
- GPU gates 在任何 fresh GPU result 产生前写入 `GPU_CONFIRM_PROTOCOL.json`
- 本 GPU runner 不会调用 selection/certification 代码生成新 context。

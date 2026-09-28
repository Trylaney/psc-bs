# v2.7 Fresh CPU confirmation 数据准备

把以下 **从未用于 v2.7 development** 的原始文件放进 `incoming/`，不要改文件名。

ACS 2023 1-Year PUMS（美国 Census 官方）：

- `csv_pco.zip` — Colorado
- `csv_pmi.zip` — Michigan
- `csv_pmn.zip` — Minnesota
- `csv_pnj.zip` — New Jersey
- `csv_por.zip` — Oregon

目录：`https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/`

另外：

- `law_dataset.csv` — `https://raw.githubusercontent.com/damtharvey/law-school-dataset/main/law_dataset.csv`
- `diabetes_130.zip` — UCI Diabetes 130-US Hospitals, dataset 296（下载页面：https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008）

放好后：

```bash
python PREPARE_FRESH_DATA.py
python v27_fresh_cpu.py
```

主方法已经冻结。**fresh 结果出来后不得改阈值、模型集合、状态集合、预算或 gates 再把同一批数据叫 confirmation。**

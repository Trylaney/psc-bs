from pathlib import Path
import zipfile,shutil,sys
root=Path(__file__).resolve().parent; incoming=root/'incoming'; cache=root/'data_cache'; dest=cache/'folktables'/'2023'/'1-Year';dest.mkdir(parents=True,exist_ok=True)
states={'CO':'08','MI':'26','MN':'27','NJ':'34','OR':'41'}
for st,code in states.items():
    z=incoming/f'csv_p{st.lower()}.zip'; target=dest/f'psam_p{code}.csv'
    if target.exists():print('OK',target);continue
    if not z.exists():print('MISSING',z);continue
    with zipfile.ZipFile(z) as zz:
        hits=[n for n in zz.namelist() if Path(n).name.lower()==target.name.lower()]
        if len(hits)!=1:raise RuntimeError((z,hits[:10]))
        with zz.open(hits[0]) as src,open(target,'wb') as out:shutil.copyfileobj(src,out)
    print('EXTRACTED',target)
law=incoming/'law_dataset.csv'
if law.exists() and not (cache/'law_dataset.csv').exists():shutil.copy2(law,cache/'law_dataset.csv');print('COPIED law_dataset.csv')
di=incoming/'diabetes_130.zip'
if di.exists() and not (cache/'diabetes_130.zip').exists():shutil.copy2(di,cache/'diabetes_130.zip');print('COPIED diabetes_130.zip')

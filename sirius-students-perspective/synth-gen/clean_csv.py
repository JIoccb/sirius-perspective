import pandas as pd
from pathlib import Path
import re

DS_PATH = Path('/home/temmie/cdng/gpb/sirius-students-perspective/synth-generator/perp')
CSV_PATH = DS_PATH / 'annotations.csv'

with open(CSV_PATH, 'r') as f:
    lines = f.readlines()

fixed_lines = [lines[0]]  

for line in lines[1:]:
    line = re.sub(r'(\[\[.*?\]\])', r'"\1"', line)
    line = re.sub(r'(\[[\d\s,]+\])', r'"\1"', line)
    fixed_lines.append(line)

with open(CSV_PATH, 'w') as f:
    f.writelines(fixed_lines)

df = pd.read_csv(CSV_PATH)
print(df.dtypes)
print(df.head())

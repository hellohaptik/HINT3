import sys
import pathlib
import pandas as pd

in_path = sys.argv[1]
p = pathlib.Path(in_path)
print(p)

df = pd.read_csv(str(p))
data = []
for k, g_df in df.groupby('label'):
    data.append({'node_name': k, 'question': '|'.join(g_df['sentence']), 'answer': f'Answer for {k}'})
out_df = pd.DataFrame(data)
out_df.to_csv(p.name, index=False)

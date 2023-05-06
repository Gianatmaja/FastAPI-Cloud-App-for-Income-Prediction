import pandas as pd

df = pd.read_csv('Performance_on_slices.csv')

#specify path for export
path = 'slice_output.txt'

#export DataFrame to text file
with open(path, 'a') as f:
    df_string = df.to_string(header=True, index=False)
    f.write(df_string)
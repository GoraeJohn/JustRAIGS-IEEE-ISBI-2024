import os
import numpy as np
import pandas as pd

csv_path = "D:\\JustRAIGS_revised_pair_removed.csv"
df = pd.read_csv(csv_path)

df_ID = df['Eye ID']
df_FellowID = df['Fellow Eye ID']
df_gt = df['gt_binary']

side_flag = 0
drop_list = []
check_list = np.zeros((len(df_ID)), dtype=np.int)

for idx1, id in enumerate(df_ID):
    print(idx1)
    if df_FellowID[idx1].__class__ == float or df_gt[idx1] == 1:
        continue
    for idx2, fellow_id in enumerate(df_FellowID):
        if fellow_id == id:
            if df_gt[idx1] == df_gt[idx2] and check_list[idx1] == 0 and check_list[idx2] == 0:
                if side_flag == 0:
                    drop_list.append(idx1)
                    check_list[idx1] = 1
                    check_list[idx2] = 1
                    side_flag = 1
                else:
                    drop_list.append(idx2)
                    check_list[idx1] = 1
                    check_list[idx2] = 1
                    side_flag = 0
            break

df.drop(drop_list, axis=0, inplace=True)
df.to_csv("D:\JustRAIGS_revised_pair_removed_final.csv")

print('Finished')
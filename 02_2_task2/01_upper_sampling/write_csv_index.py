import albumentations as A
import cv2
import numpy as np
import os
import pandas as pd

IS_SAVE = True
IS_DEBUG = False

# POST_FIX = 'upper_v_flip'
# POST_FIX = 'upper_v_mirror' 
# POST_FIX = 'upper_v_flip_mirror' 
#POST_FIX = 'upper_v_flip_disc_copy'
POST_FIX = 'upper_v_flip_mirror_disc_copy'
# POST_FIX = 'upper_v_flip_disc_copy_flip'
# POST_FIX = 'upper_v_flip_mirror_disc_copy_flip'
# INPUT_CSV_PATH = "/root/home/challenge/justraigs/binary_label/label_info_with_test/JustRAIGS_Train_labels_segment_from_G3_revised_pair_removed_final_train_val.csv" ###
# INPUT_CSV_PATH = "/root/home/challenge/justraigs/multi_label/label_info/JustRAIGS_Train_labels_multi_with_normal_from_G3.csv"
INPUT_CSV_PATH = "/root/home/challenge/justraigs/multi_label/label_info/JustRAIGS_Train_labels_multi_from_G3.csv"
# INPUT_CSV_PATH = "/root/home/challenge/justraigs/multi_label/label_info/JustRAIGS_Train_labels_segment_from_G3_removed_pair.csv"

# UPPER_CSV_PATH = ["/root/home/challenge/justraigs/multi_label/test_upper_sampling/upper_v_flip.csv"]
# UPPER_POST_FIX = ['v_f']
# UPPER_CSV_PATH = ["/root/home/challenge/justraigs/multi_label/test_upper_sampling/upper_v_m_s.csv",
#                   "/root/home/challenge/justraigs/multi_label/test_upper_sampling/upper_v_m_i.csv"]
# UPPER_POST_FIX = ['v_m_s',
#                   'v_m_i']
# UPPER_CSV_PATH = ["/root/home/challenge/justraigs/multi_label/test_upper_sampling/upper_v_flip.csv",
#                   "/root/home/challenge/justraigs/multi_label/test_upper_sampling/upper_v_m_s.csv",
#                   "/root/home/challenge/justraigs/multi_label/test_upper_sampling/upper_v_m_i.csv"]
# UPPER_POST_FIX = ['v_f',
#                   'v_m_s',
#                   'v_m_i']
# UPPER_CSV_PATH = ["/root/home/challenge/justraigs/test_upper_sampling/upper_v_flip.csv",                  
#                   '/root/home/challenge/justraigs/test_upper_sampling/upper_disc_copy.csv']
# UPPER_POST_FIX = ['v_f',                  
#                   'disc_copy']

# UPPER_CSV_PATH = ["/root/home/challenge/justraigs/test_upper_sampling/upper_v_flip.csv",                  
#                   '/root/home/challenge/justraigs/test_upper_sampling/upper_disc_copy.csv',
#                   '/root/home/challenge/justraigs/test_upper_sampling/upper_disc_copy_v_f.csv']
# UPPER_POST_FIX = ['v_f',                  
#                   'disc_copy',
#                   'disc_copy_disc_copy_v_f']

# UPPER_CSV_PATH = ["/root/home/challenge/justraigs/test_upper_sampling/upper_v_flip.csv", 
#                   "/root/home/challenge/justraigs/test_upper_sampling/upper_v_m_s.csv",
#                   "/root/home/challenge/justraigs/test_upper_sampling/upper_v_m_i.csv",                 
#                   '/root/home/challenge/justraigs/test_upper_sampling/upper_disc_copy.csv',
#                   '/root/home/challenge/justraigs/test_upper_sampling/upper_disc_copy_v_f.csv']
# UPPER_POST_FIX = ['v_f',
#                   'v_m_s',
#                   'v_m_i',                  
#                   'disc_copy',
#                   'disc_copy_disc_copy_v_f']

UPPER_CSV_PATH = ["/root/home/challenge/justraigs/test_upper_sampling/upper_v_flip.csv", 
                  "/root/home/challenge/justraigs/test_upper_sampling/upper_v_m_s.csv",
                  "/root/home/challenge/justraigs/test_upper_sampling/upper_v_m_i.csv",                 
                  '/root/home/challenge/justraigs/test_upper_sampling/upper_disc_copy.csv']
UPPER_POST_FIX = ['v_f',
                  'v_m_s',
                  'v_m_i',                  
                  'disc_copy']

OUTPUT_CSV_PATH = INPUT_CSV_PATH.replace(".csv", f"_{POST_FIX}.csv")

INDEX_PATH = '/root/home/challenge/justraigs/multi_label/index_list'
INDEX_PREFIX = 'multi_labeled'
# INDEX_PREFIX = 'multi_labeled_with_normal'
# INDEX_PREFIX = 'multi_labeled_for_all'
# INDEX_PREFIX = 'multi_labeled_remove_pair'
INDEX_PREFIX_TRAIN = f'{INDEX_PREFIX}_train_index'
# INDEX_PREFIX_TRAIN = f'train_index'

CSV_ID_IMAGE_NAME = 'Eye ID'

input_df = pd.read_csv(INPUT_CSV_PATH)
input_file_name_list = np.array(input_df[CSV_ID_IMAGE_NAME].values)

output_df = input_df.copy()
for current_csv in UPPER_CSV_PATH:
    current_df = pd.read_csv(current_csv)
    output_df = pd.concat((output_df, current_df))
    
output_file_name_list = np.array(output_df[CSV_ID_IMAGE_NAME].values)

for i in range(5):
    train_index_file_fath = os.path.join(INDEX_PATH, f'{INDEX_PREFIX_TRAIN}_{i}.npy')
    train_index = np.load(train_index_file_fath)
    result_train_index = train_index.tolist()
    
    for index in train_index:
        current_file_name = input_file_name_list[index] 
        
        for current_upper_post_fix in UPPER_POST_FIX:
        
            current_upper_sample_name = f'{current_file_name}_{current_upper_post_fix}' 
            
            current_upper_sample_index = np.where(output_file_name_list == current_upper_sample_name)[0]
            if len(current_upper_sample_index) > 0:
                result_train_index.append(current_upper_sample_index[0])
    
    output_train_index_file_fath = os.path.join(INDEX_PATH, f'{INDEX_PREFIX_TRAIN}_{POST_FIX}_{i}.npy')
    np.save(output_train_index_file_fath, result_train_index)

output_df.to_csv(OUTPUT_CSV_PATH, index=False)
print("finish")
    
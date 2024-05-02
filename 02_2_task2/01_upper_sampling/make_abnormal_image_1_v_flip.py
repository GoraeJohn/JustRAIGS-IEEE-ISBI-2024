import albumentations as A
import cv2
import numpy as np
import os
import pandas as pd

IS_SAVE = True
IS_DEBUG = False
POST_FIX = 'v_f' #Data + POST_FIX

INPUT_CSV_PATH = "/root/home/challenge/justraigs/multi_label/label_info/JustRAIGS_Train_labels_multi_from_G3.csv" ###
OUTPUT_CSV_PATH = INPUT_CSV_PATH.replace(".csv", "_upper_v_flip.csv")

INDEX_PATH = '/root/home/challenge/justraigs/multi_label/index_list'
INDEX_PREFIX = 'multi_labeled'
INDEX_PREFIX_TRAIN = f'{INDEX_PREFIX}_train_index'

INPUT_PATH = "/root/home/challenge/justraigs/train_data_preprocess_npy"
OUTPUT_PATH = "/root/home/challenge/justraigs/train_data_preprocess_npy"
#OUTPUT_PATH = f'{INPUT_PATH}_{POST_FIX}'

DEBUG_INPUT_PATH = "/root/home/challenge/justraigs/multi_label/test_upper_sampling/debug_input" #Input Image (original)
DEBUG_OUTPUT_PATH = "/root/home/challenge/justraigs/multi_label/test_upper_sampling/debug_output" #Output Image (processed)


def transform_gt_label(input_label, swap_id):
    result_label = [0] * len(input_label)
    if len(input_label) == len(swap_id):
        for index in range(len(input_label)):
            result_label[index] = input_label[swap_id[index]]
            
    return result_label

def save_image_file(input_file, input_data):
    cv2.imwrite(input_file, input_data)
#Load 2048x2048 Image
#Transform
#Generate CSV and Indexing (Train index ) ... 여러가지 csv/index...

train_transform =  A.Compose([        
        A.VerticalFlip(p=1),                
    ])

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
if IS_DEBUG:
    if not os.path.exists(DEBUG_INPUT_PATH):
        os.makedirs(DEBUG_INPUT_PATH)
    if not os.path.exists(DEBUG_OUTPUT_PATH):
        os.makedirs(DEBUG_OUTPUT_PATH)
    
#From CSV...
LABEL_ID = ['ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']
SWAP_ID = [1, 0, 3, 2, 5, 4, 6, 7, 8, 9]

CSV_ID_PREFIX_GT = 'gt_'
CSV_ID_PREFIX_GT1 = 'G1 '
CSV_ID_PREFIX_GT2 = 'G2 '
CSV_ID_PREFIX_GT3 = 'G3 '

CSV_ID_IMAGE_NAME = 'Eye ID'
CSV_ID_EXT = 'image_ext'
CSV_ID_GT_BINARY = 'gt_binary'

CSV_ID_GT_LABEL = []
CSV_ID_G1_LABEL = []
CSV_ID_G2_LABEL = []
CSV_ID_G3_LABEL = []
for current_label in LABEL_ID:
    CSV_ID_GT_LABEL.append(CSV_ID_PREFIX_GT + current_label)
    CSV_ID_G1_LABEL.append(CSV_ID_PREFIX_GT1 + current_label)
    CSV_ID_G2_LABEL.append(CSV_ID_PREFIX_GT2 + current_label)
    CSV_ID_G3_LABEL.append(CSV_ID_PREFIX_GT3 + current_label)

df_val = pd.read_csv(INPUT_CSV_PATH)
df_val_gt = df_val['gt_binary']
## 대상이미지는 gt_binary가 1인 것들만...

image_file_list = df_val[CSV_ID_IMAGE_NAME].values
gt_binary_list = df_val[CSV_ID_GT_BINARY].values
gt_label_list = df_val[CSV_ID_GT_LABEL]

target_index = np.where(gt_binary_list == 1)[0]

output_raws = []

for current_index in target_index:
    current_raw = df_val.iloc[current_index]
    
    current_file_name = current_raw[CSV_ID_IMAGE_NAME]    
    current_gt_label = current_raw[CSV_ID_GT_LABEL].values
    if np.sum(current_gt_label) <= 0:
        continue    
    #Load current image
    #transform image
    
    current_input_image_path = os.path.join(INPUT_PATH, f'{current_file_name}.npy')
    current_output_name = f'{current_file_name}_{POST_FIX}'    
    current_output_image_path = os.path.join(OUTPUT_PATH, f'{current_output_name}.npy')
    
    if IS_SAVE:
        current_input_image = np.load(current_input_image_path)    
        res = train_transform(image=current_input_image)
        current_output_image = res['image']    
        np.save(current_output_image_path, current_output_image)
        
    current_output_gt_label = transform_gt_label(current_gt_label, SWAP_ID)    
    current_raw[CSV_ID_GT_LABEL] = current_output_gt_label
    current_raw[CSV_ID_G1_LABEL] = transform_gt_label(current_raw[CSV_ID_G1_LABEL].values, SWAP_ID)
    current_raw[CSV_ID_G2_LABEL] = transform_gt_label(current_raw[CSV_ID_G2_LABEL].values, SWAP_ID)
    current_raw[CSV_ID_G3_LABEL] = transform_gt_label(current_raw[CSV_ID_G3_LABEL].values, SWAP_ID)
    current_raw[CSV_ID_IMAGE_NAME] = current_output_name
    
    output_raws.append(current_raw)
    if IS_DEBUG:
        debug_input_file_name = os.path.join(DEBUG_INPUT_PATH, f'{current_file_name}.jpg')
        debug_output_file_name = os.path.join(DEBUG_OUTPUT_PATH, f'{current_output_name}.jpg')
        save_image_file(debug_input_file_name, current_input_image)
        save_image_file(debug_output_file_name, current_output_image)

output_df = pd.DataFrame(output_raws)
output_df.columns = df_val.columns
output_df = pd.concat((df_val, output_df))
output_df.to_csv(OUTPUT_CSV_PATH, index=False)

output_file_name_list = np.array(output_df[CSV_ID_IMAGE_NAME].values)
for i in range(5):
    train_index_file_fath = os.path.join(INDEX_PATH, f'{INDEX_PREFIX_TRAIN}_{i}.npy')
    train_index = np.load(train_index_file_fath)
    result_train_index = train_index.tolist()
    ##
    for index in train_index:
        current_file_name = df_val.iloc[index][CSV_ID_IMAGE_NAME] 
        current_upper_sample_name = f'{current_file_name}_{POST_FIX}' 
        #이것의 index를 찾아서 기존 train_index에 append하자..
        current_upper_sample_index = np.where(output_file_name_list == current_upper_sample_name)[0]
        if len(current_upper_sample_index) > 0:
            result_train_index.append(current_upper_sample_index[0])
    
    output_train_index_file_fath = os.path.join(INDEX_PATH, f'{INDEX_PREFIX_TRAIN}_{POST_FIX}_{i}.npy')
    np.save(output_train_index_file_fath, result_train_index)
        
print("finish")
    

    
import albumentations as A
import cv2
import numpy as np
import os
import pandas as pd

IS_SAVE = True
IS_DEBUG = True
POST_FIX = 'disc_copy' #Data + POST_FIX

INPUT_CSV_PATH = "/root/home/challenge/justraigs/multi_label/label_info/JustRAIGS_Train_labels_multi_from_G3.csv" ###
REFERENCE_CSV_PATH = "/root/home/challenge/justraigs/multi_label/label_info/JustRAIGS_Train_labels_segment_from_G3.csv"
OUTPUT_CSV_PATH = "/root/home/challenge/justraigs/test_upper_sampling/upper_disc_copy.csv"

INDEX_PATH = '/root/home/challenge/justraigs/multi_label/index_list'
INDEX_PREFIX = 'multi_labeled'
INDEX_PREFIX_TRAIN = f'{INDEX_PREFIX}_train_index'

INPUT_PATH = "/root/home/challenge/justraigs/train_data_preprocess_npy"
#OUTPUT_PATH = "/root/home/challenge/justraigs/train_data_preprocess_npy"
OUTPUT_PATH = f'{INPUT_PATH}_{POST_FIX}'

DEBUG_INPUT_PATH = f"/root/home/challenge/justraigs/test_upper_sampling/debug_input_{POST_FIX}" #Input Image (original)
DEBUG_OUTPUT_PATH = f"/root/home/challenge/justraigs/test_upper_sampling/debug_output_{POST_FIX}" #Output Image (processed)

def save_image_file(input_file, input_data):
    cv2.imwrite(input_file, input_data)
    
def check_valid_image(gt_list, valid_gt_list):
    if np.sum(gt_list) <= 0:
        return False
    
    for index, current_gt_label in enumerate(gt_list):
        if valid_gt_list[index] == True and current_gt_label == 1:
            return True
    return False    

def transform_gt_label(input_label, reference_label, valid_gt_list):
    result_label = [0] * len(input_label)
    if len(input_label) == len(valid_gt_list):
        for index in range(len(input_label)):
            if valid_gt_list[index]:
                result_label[index] = input_label[index]
            else:
                result_label[index] = reference_label[index]
            
    return result_label

def convert_rect_to_index(input_rect, image_width, image_height, margin=0):
    x_min = max(0, input_rect[0] - margin)
    y_min = max(0, input_rect[1] - margin)
    x_max = min(image_width, x_min + input_rect[2] + margin)
    y_max = min(image_height, y_min + input_rect[3] + margin)
    
    return x_min, y_min, x_max, y_max
    
def copy_disc_image(input_image, input_rect, reference_image, reference_rect):
    """ rect = [x, y, w, h] """    
    result_image = np.zeros(reference_image.shape, reference_image.dtype)
    
    input_x_min, input_y_min, input_x_max, input_y_max = convert_rect_to_index(input_rect, input_image.shape[1], input_image.shape[0])
    reference_x_min, reference_y_min, reference_x_max, reference_y_max = convert_rect_to_index(reference_rect, reference_image.shape[1], reference_image.shape[0])
    
    input_crop = input_image[input_y_min:input_y_max, input_x_min:input_x_max, :]
    input_crop_resized = cv2.resize(input_crop, (reference_x_max - reference_x_min, reference_y_max - reference_y_min), cv2.INTER_AREA)
    input_crop_resized_flip = cv2.flip(input_crop_resized, 1)
    result_image[:, :, :] = reference_image[:, :, :]
    result_image[reference_y_min:reference_y_max, reference_x_min:reference_x_max, :] = input_crop_resized_flip
    
    return result_image
#Load 2048x2048 Image
#Transform
#Generate CSV and Indexing (Train index ) ... 여러가지 csv/index...

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
if IS_DEBUG:
    if not os.path.exists(DEBUG_INPUT_PATH):
        os.makedirs(DEBUG_INPUT_PATH)
    if not os.path.exists(DEBUG_OUTPUT_PATH):
        os.makedirs(DEBUG_OUTPUT_PATH)
    
#From CSV...
# 	        0	    0.5	    1	    Total	inside DISC
# ANRS	    4161	488	    2021	6670	O
# ANRI	    3942	477	    2251	6670	O
# RNFLDS	6114	364	    192	    6670	X
# RNFLDI	5999	442	    229	    6670	X
# BCLVS	    5623	717	    330	    6670	O
# BCLVI	    5541	789	    340	    6670	O
# NVT	    5212	941	    517	    6670	O
# DH	    6527	61	    82	    6670	O
# LD	    5037	779	    854	    6670	O
# LC	    4524	1008	1138	6670	O
LABEL_ID = ['ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']
COPY_ID = [False, False, False, False, True, True, False, True, False, False]
LABEL_ID_VALID = [True, True, False, False, True, True, True, True, True, True]
CSV_RECT_ID = ['rect_disc_x', 'rect_disc_y', 'rect_disc_w', 'rect_disc_h']

CSV_ID_PREFIX_GT = 'gt_'
CSV_ID_PREFIX_GT1 = 'G1 '
CSV_ID_PREFIX_GT2 = 'G2 '
CSV_ID_PREFIX_GT3 = 'G3 '

CSV_ID_IMAGE_NAME = 'Eye ID' 
CSV_ID_IMAGE_TARGET_NAME = 'Fellow Eye ID'

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

gt_binary_list = df_val[CSV_ID_GT_BINARY].values
target_index = np.where(gt_binary_list == 1)[0]

reference_df = pd.read_csv(REFERENCE_CSV_PATH)
reference_image_file_list = np.array(reference_df[CSV_ID_IMAGE_NAME].values)

output_raws = []

for current_index in target_index:
    current_raw = df_val.iloc[current_index]
    
    current_file_name = current_raw[CSV_ID_IMAGE_NAME]    
    current_target_name = current_raw[CSV_ID_IMAGE_TARGET_NAME]    
    current_gt_label = current_raw[CSV_ID_GT_LABEL].values
    current_rect = current_raw[CSV_RECT_ID].values
        
    if check_valid_image(current_gt_label, COPY_ID) == False:
        continue    
    
    reference_index = np.where(reference_image_file_list == current_target_name)[0]
    if len(reference_index) <= 0: 
        continue
    
    reference_raw = reference_df.iloc[reference_index[0]]
    
    reference_rect = reference_raw[CSV_RECT_ID].values
        
    #Load current image
    #transform image
    
    current_input_image_path = os.path.join(INPUT_PATH, f'{current_file_name}.npy')
    current_output_name = f'{current_file_name}_{POST_FIX}'    
    current_output_image_path = os.path.join(OUTPUT_PATH, f'{current_output_name}.npy')
    
    current_target_image_path  = os.path.join(INPUT_PATH, f'{current_target_name}.npy')
    
    curreng_reference_gt_label = reference_raw[CSV_ID_GT_LABEL].values  
    current_output_gt_label = transform_gt_label(current_gt_label, curreng_reference_gt_label, LABEL_ID_VALID)   
     
    reference_raw[CSV_ID_GT_LABEL] = current_output_gt_label    
    reference_raw[CSV_ID_IMAGE_NAME] = current_output_name
    
    
    output_raws.append(reference_raw)
    
    if IS_SAVE:
        current_input_image = np.load(current_input_image_path)    
        current_target_image = np.load(current_target_image_path)
        
        current_output_image = copy_disc_image(current_input_image, current_rect, current_target_image, reference_rect)
        np.save(current_output_image_path, current_output_image)    
    
        if IS_DEBUG:
            debug_input_file_name = os.path.join(DEBUG_INPUT_PATH, f'{current_file_name}.jpg')
            debug_output_file_name = os.path.join(DEBUG_OUTPUT_PATH, f'{current_output_name}.jpg')
            save_image_file(debug_input_file_name, current_input_image)
            save_image_file(debug_output_file_name, current_output_image)

output_df = pd.DataFrame(output_raws)
output_df.columns = df_val.columns
output_df.to_csv(OUTPUT_CSV_PATH, index=False)

print("finish")
    

    
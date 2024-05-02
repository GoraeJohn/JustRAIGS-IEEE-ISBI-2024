import os
import cv2 as cv
from resources.pre_processor import Processor

INPUT_IMAGE_PATH = '/root/justraigs/train_data'
PREPROCESS_IMAGE_PATH = '/root/justraigs/train_data_preprocess'

MODEL_IMAGE_SIZE = 2048

def check_make_path(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)

check_make_path(PREPROCESS_IMAGE_PATH)
processor = Processor(True, MODEL_IMAGE_SIZE)

input_image_list = os.listdir(INPUT_IMAGE_PATH)
for current_image_name in input_image_list:    
    current_image_path = os.path.join(INPUT_IMAGE_PATH, current_image_name)
    
    current_image = cv.imread(current_image_path, cv.IMREAD_COLOR)    
    if current_image is None:
        continue
    if len(current_image) <= 0:
        continue
    
    processed_image = processor.process(current_image) 
    #Need crop, resize info or with mask
    cv.imwrite(os.path.join(PREPROCESS_IMAGE_PATH, current_image_name), processed_image)
    print('run: ', current_image_name)
    
print("finish")
    
        


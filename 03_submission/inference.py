import random

import numpy
from PIL import Image
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks
from resources.pre_processor import Processor
from resources.run_model import RunTorch
import os
import cv2 as cv    
import numpy as np

RAW_PATH = './' #for submission

IS_DEBUG = True
if IS_DEBUG:
    RAW_PATH = '/root/justraigs/JustRAIGS-IEEE-ISBI-2024/kt_algorithm6' #for debugging
    debug_path = '/root/justraigs/JustRAIGS-IEEE-ISBI-2024/kt_algorithm6/test'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "0" #GPU ID
            
IMAGE_SIZE = 512
THRESHOLD_TASK1 = 0.087663
THRESHOLD_TASK2 = [0.5353535353535354, 0.5555555555555556, 0.595959595959596, 0.6262626262626263, 0.6666666666666667, 0.6363636363636365, 0.5353535353535354, 0.4646464646464647, 0.5252525252525253, 0.5050505050505051]
MODEL_FILE_LIST_TASK1 = []
MODEL_FILE_LIST_TASK2_HIGH = []
MODEL_FILE_LIST_TASK2_LOW = []
MODEL_FILE_LIST_TASK2_REFERENCE = []

TASK2_INDEX_HIGH = [0, 1, 6, 8, 9]
TASK2_INDEX_LOW = [2, 3, 4, 5, 7]
    
def run():
    _show_torch_cuda_info()
    
    for index in range(5):
        MODEL_FILE_LIST_TASK1.append(os.path.join(RAW_PATH, "resources", "model_task1", f'model_sen_{index}.pth'))
        MODEL_FILE_LIST_TASK2_HIGH.append(os.path.join(RAW_PATH, "resources", "model_task2", "model_high", f'model_auc_fold_{index}.pth'))
        MODEL_FILE_LIST_TASK2_LOW.append(os.path.join(RAW_PATH, "resources", "model_task2", "model_low", f'model_auc_fold_{index}.pth'))
        MODEL_FILE_LIST_TASK2_REFERENCE.append(os.path.join(RAW_PATH, "resources", "model_task2", "model_reference", f'model_auc_fold_{index}.pth'))
        
    preprocessor = Processor(True, IMAGE_SIZE)
    run_task1 = RunTorch(MODEL_FILE_LIST_TASK1, out_num=1, model_version=2, is_sigmoid=True)        
    run_task2_high = RunTorch(MODEL_FILE_LIST_TASK2_HIGH, out_num=len(TASK2_INDEX_HIGH), model_version=2, is_sigmoid=True)    
    run_task2_low = RunTorch(MODEL_FILE_LIST_TASK2_LOW, out_num=len(TASK2_INDEX_LOW), model_version=2, is_sigmoid=True)    
    run_task_2_reference = RunTorch(MODEL_FILE_LIST_TASK2_REFERENCE, out_num=len(TASK2_INDEX_HIGH)+len(TASK2_INDEX_LOW), model_version=2, is_sigmoid=True)    
    
    for jpg_image_file_name, save_prediction in inference_tasks():
        # Do inference, possibly something better performant
        print(f"Running inference on {jpg_image_file_name}")

        # Image load and Preprosessing: 0~1.0 [1, 3, IMAGE_SIZE, IMAGE_SIZE]
        # image = Image.open(jpg_image_file_name)        
        # numpy_array = numpy.array(image)
        
        numpy_array = cv.imread(jpg_image_file_name._str) #BGR        
        processed_image = preprocessor.process(numpy_array) #RGB
        
        if IS_DEBUG:            
            print(f"BGR: 0 channel: {np.mean(numpy_array[:, :, 0])}, 2 channel {np.mean(numpy_array[:, :, 2])}")        
            print(f"RGB: 0 channel: {np.mean(processed_image[0, 0, :, :])}, 2 channel {np.mean(processed_image[0, 2, :, :,])}")        
            cv.imwrite(os.path.join(debug_path, "input_" + jpg_image_file_name.name), numpy_array)            
        
        is_referable_glaucoma_likelihood = run_task1.run_single_batch(processed_image)    
        features_high = run_task2_high.run_single_batch(processed_image)             
        features_low = run_task2_low.run_single_batch(processed_image)             
        features_select = np.zeros(len(TASK2_INDEX_HIGH) + len(TASK2_INDEX_LOW))
        features_select[TASK2_INDEX_HIGH] = features_high
        features_select[TASK2_INDEX_LOW] = features_low
        
        features_reference = run_task_2_reference.run_single_batch(processed_image)
        features = 0.5 * features_select + 0.5 * np.array(features_reference)
        features = features.tolist()
        
        if IS_DEBUG:
            print(is_referable_glaucoma_likelihood)
            print(features)
            
        is_referable_glaucoma = is_referable_glaucoma_likelihood > THRESHOLD_TASK1
                            
        feature_dict = {}
        for index_key, current_key in enumerate(DEFAULT_GLAUCOMATOUS_FEATURES.keys()):
            feature_dict[current_key] = features[index_key] > THRESHOLD_TASK2[index_key]
                    
        # Finally, save the answer
        save_prediction(
            is_referable_glaucoma,
            is_referable_glaucoma_likelihood,
            feature_dict,
        )
    return 0


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())

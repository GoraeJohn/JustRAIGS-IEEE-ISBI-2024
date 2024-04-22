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

IS_DEBUG = False
if IS_DEBUG:
    RAW_PATH = '/root/challenge/justraigs/submission_files/kt_algorithm_final' #for debugging
    debug_path = '/root/challenge/justraigs/submission_files/kt_algorithm_final/test'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "0" #GPU ID
            
IMAGE_SIZE = 512
THRESHOLD_TASK1_B8 = 0.0703612
THRESHOLD_TASK1_B64 = 0.087663
THRESHOLD_TASK1_ENSEMBLE = 0.5 * THRESHOLD_TASK1_B8 + 0.5 * THRESHOLD_TASK1_B64
THRESHOLD_TASK2 = [0.5454545454545455, 0.5252525252525253, 0.5252525252525253, 0.5858585858585859, 0.6767676767676768, 0.5858585858585859, 0.5151515151515152, 0.42424242424242425, 0.5858585858585859, 0.5050505050505051]

MODEL_FILE_LIST_TASK1_B8 = []
MODEL_FILE_LIST_TASK1_B64 = []
MODEL_FILE_LIST_TASK2_HIGH = []
MODEL_FILE_LIST_TASK2_LOW = []
MODEL_FILE_LIST_TASK2_REFERENCE = []
MODEL_FILE_LIST_TASK2_2_HIGH = []
MODEL_FILE_LIST_TASK2_2_LOW = []
MODEL_FILE_LIST_TASK2_2_REFERENCE = []

TASK2_INDEX_HIGH = [0, 1, 6, 8, 9]
TASK2_INDEX_LOW = [2, 3, 4, 5, 7]
    
def run():
    _show_torch_cuda_info()
    
    for index in range(5):
        MODEL_FILE_LIST_TASK1_B8.append(os.path.join(RAW_PATH, "resources", "model_task1", "model_b8", f'model_auc_{index}.pth'))
        MODEL_FILE_LIST_TASK1_B64.append(os.path.join(RAW_PATH, "resources", "model_task1", "model_b64", f'model_auc_{index}.pth'))
        MODEL_FILE_LIST_TASK2_HIGH.append(os.path.join(RAW_PATH, "resources", "model_task2_b32", "model_high", f'model_auc_fold_{index}.pth'))
        MODEL_FILE_LIST_TASK2_LOW.append(os.path.join(RAW_PATH, "resources", "model_task2_b32", "model_low", f'model_auc_fold_{index}.pth'))
        MODEL_FILE_LIST_TASK2_REFERENCE.append(os.path.join(RAW_PATH, "resources", "model_task2_b32", "model_reference", f'model_auc_fold_{index}.pth'))
        
        MODEL_FILE_LIST_TASK2_2_HIGH.append(os.path.join(RAW_PATH, "resources", "model_task2_b32_2", "model_high", f'model_auc_fold_{index}.pth'))
        MODEL_FILE_LIST_TASK2_2_LOW.append(os.path.join(RAW_PATH, "resources", "model_task2_b32_2", "model_low", f'model_auc_fold_{index}.pth'))
        MODEL_FILE_LIST_TASK2_2_REFERENCE.append(os.path.join(RAW_PATH, "resources", "model_task2_b32_2", "model_reference", f'model_auc_fold_{index}.pth'))
        
    preprocessor = Processor(True, IMAGE_SIZE)
    run_task1_b8 = RunTorch(MODEL_FILE_LIST_TASK1_B8, out_num=1, model_version=2, is_sigmoid=True)
    run_task1_b64 = RunTorch(MODEL_FILE_LIST_TASK1_B64, out_num=1, model_version=2, is_sigmoid=True)        
    run_task2_high = RunTorch(MODEL_FILE_LIST_TASK2_HIGH, out_num=len(TASK2_INDEX_HIGH), model_version=2, is_sigmoid=True)    
    run_task2_low = RunTorch(MODEL_FILE_LIST_TASK2_LOW, out_num=len(TASK2_INDEX_LOW), model_version=2, is_sigmoid=True)    
    run_task_2_reference = RunTorch(MODEL_FILE_LIST_TASK2_REFERENCE, out_num=len(TASK2_INDEX_HIGH)+len(TASK2_INDEX_LOW), model_version=2, is_sigmoid=True)    
    run_task2_2_high = RunTorch(MODEL_FILE_LIST_TASK2_2_HIGH, out_num=len(TASK2_INDEX_HIGH), model_version=2, is_sigmoid=True)    
    run_task2_2_low = RunTorch(MODEL_FILE_LIST_TASK2_2_LOW, out_num=len(TASK2_INDEX_LOW), model_version=2, is_sigmoid=True)    
    run_task_2_2_reference = RunTorch(MODEL_FILE_LIST_TASK2_2_REFERENCE, out_num=len(TASK2_INDEX_HIGH)+len(TASK2_INDEX_LOW), model_version=2, is_sigmoid=True)    
    
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
        
        task1_prob_b8 = run_task1_b8.run_single_batch(processed_image)
        task1_prob_b64 = run_task1_b64.run_single_batch(processed_image)
        features_high = run_task2_high.run_single_batch(processed_image)             
        features_low = run_task2_low.run_single_batch(processed_image)             
        features_select = np.zeros(len(TASK2_INDEX_HIGH) + len(TASK2_INDEX_LOW))
        features_select[TASK2_INDEX_HIGH] = features_high
        features_select[TASK2_INDEX_LOW] = features_low
        
        features_reference = run_task_2_reference.run_single_batch(processed_image)
        features = 0.5 * features_select + 0.5 * np.array(features_reference)
        
        features_2_high = run_task2_2_high.run_single_batch(processed_image)             
        features_2_low = run_task2_2_low.run_single_batch(processed_image)             
        features_2_select = np.zeros(len(TASK2_INDEX_HIGH) + len(TASK2_INDEX_LOW))
        features_2_select[TASK2_INDEX_HIGH] = features_2_high
        features_2_select[TASK2_INDEX_LOW] = features_2_low
        
        features_reference_2 = run_task_2_2_reference.run_single_batch(processed_image)
        features_2 = 0.5 * features_2_select + 0.5 * np.array(features_reference_2)
        
        features = 0.5 * features + 0.5 * features_2
        features = features.tolist()
        
        is_referable_glaucoma_likelihood = 0.5 * task1_prob_b8 + 0.5 * task1_prob_b64
        
        if IS_DEBUG:
            print(is_referable_glaucoma_likelihood, task1_prob_b8, task1_prob_b64)
            print(features)
            
        is_referable_glaucoma = is_referable_glaucoma_likelihood > THRESHOLD_TASK1_ENSEMBLE
                            
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

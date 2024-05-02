import pandas as pd
import os
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc

ORIGINAL_CSV_PATH = '/root/home/challenge/justraigs/multi_label/convnext_small_5fold_pretrained/val_convnext_small_5fold_pretrained.csv'
# REFERENCE_CSV_PATH = '/root/home/challenge/justraigs/multi_label/convnext_small_5fold_pretrained_v_flip_mirror_disc_copy_flip/val_convnext_small_5fold_pretrained_v_flip_mirror_disc_copy_flip.csv'
REFERENCE_CSV_PATH = '/root/home/challenge/justraigs/multi_label/convnext_small_5fold_pretrained_v_flip_mirror_disc_copy_flip_b4/val_convnext_small_5fold_pretrained_v_flip_mirror_disc_copy_flip_b4.csv'
TARGET_1_CSV_PATH = '/root/home/challenge/justraigs/multi_label/convnext_small_5fold_pretrained_v_flip_mirror_disc_copy_flip_high/val_convnext_small_5fold_pretrained_v_flip_mirror_disc_copy_flip_high.csv'
TARGET_2_CSV_PATH = '/root/home/challenge/justraigs/multi_label/convnext_small_5fold_pretrained_v_flip_mirror_disc_copy_flip_low/val_convnext_small_5fold_pretrained_v_flip_mirror_disc_copy_flip_low.csv'
# TARGET_1_CSV_PATH = '/root/home/challenge/justraigs/multi_label/convnext_small_5fold_pretrained_v_flip_mirror_disc_copy_flip_b4_high/val_convnext_small_5fold_pretrained_v_flip_mirror_disc_copy_flip_b4_high.csv'
# TARGET_2_CSV_PATH = '/root/home/challenge/justraigs/multi_label/convnext_small_5fold_pretrained_v_flip_mirror_disc_copy_flip_b4_low/val_convnext_small_5fold_pretrained_v_flip_mirror_disc_copy_flip_b4_low.csv'

OUTPUT_CSV_PATH_SELECT = '/root/home/challenge/justraigs/post_processing/val_output_task2_select.csv'
OUTPUT_CSV_PATH_ENSEMBLE = '/root/home/challenge/justraigs/post_processing/val_output_task2_ensemble.csv'

CSV_ID_IMAGE = 'Eye ID'

LABEL_ID = ['ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']
SELECT_NUM_TARGET_1 = [0, 1, 6, 8, 9]
SELECT_NUM_TARGET_2 = [2, 3, 4, 5, 7]

CSV_ID_PREFIX_GT = 'gt_'
CSV_ID_RESULT = []
CSV_ID_GT_LABEL = []

for current_label in LABEL_ID:
    current_gt_id = CSV_ID_PREFIX_GT + current_label
    CSV_ID_GT_LABEL.append(current_gt_id)
    
    current_result_id = "result_" + current_label
    CSV_ID_RESULT.append(current_result_id)   
   
CSV_ID_RESULT_TARGET_1 = [CSV_ID_RESULT[i] for i in SELECT_NUM_TARGET_1]
CSV_ID_RESULT_TARGET_2 = [CSV_ID_RESULT[i] for i in SELECT_NUM_TARGET_2] 

def metric_auc_sensitivity(targets, predict_sigmoid):    
    # auc = roc_auc_score(targets, predict_sigmoid)
    desired_specificity = 0.95
    fpr, tpr, thresholds = roc_curve(targets, predict_sigmoid)
    roc_auc = auc(fpr, tpr)
    
    ##
    fpr_1 = 1 - fpr
    gmeans = np.sqrt(tpr * fpr_1)
    idx = np.argmax(gmeans)
    
    #idx = np.argmax(fpr >= (1 - desired_specificity))
    threshold_best = thresholds[idx]
    sensitivity_best = tpr[idx]
    specificity_best = fpr_1[idx]
        
    return roc_auc, sensitivity_best, specificity_best, threshold_best

def metric_each_label(true_label_list, predict_label_list, valid_label_list, out_dim=len(LABEL_ID)):
    target_list = [[] for _ in range(out_dim)]
    predict_list = [[] for _ in range(out_dim)]
    for index in range(len(true_label_list)):        
        for index_label in range(out_dim):
            if valid_label_list[index][index_label]:
                predict_list[index_label].append(predict_label_list[index][index_label])
                if true_label_list[index][index_label] > 0:
                    target_list[index_label].append(1)
                else:
                    target_list[index_label].append(0)
                        
    roc_auc_list = []
    sensitivity_list = []
    specificity_list = []
    threshold_list = []
    
    for index_label in range(out_dim):
        current_roc_auc, sensitivity_best, specificity_best, threshold_best = metric_auc_sensitivity(target_list[index_label], predict_list[index_label])
        roc_auc_list.append(current_roc_auc)
        sensitivity_list.append(sensitivity_best)
        specificity_list.append(specificity_best)
        threshold_list.append(threshold_best)
        
    return roc_auc_list, sensitivity_list, specificity_list, threshold_list

def calc_hamming_loss(true_labels, predicted_labels):
    """Calculate the Hamming loss for the given true and predicted labels."""
    # Convert to numpy arrays for efficient computation
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate the hamming distance that is basically the total number of mismatches
    Hamming_distance = np.sum(np.not_equal(true_labels, predicted_labels))
        
    # Calculate the total number of labels
    total_corrected_labels= true_labels.size

    # Compute the Modified Hamming loss
    loss = Hamming_distance / total_corrected_labels
    return loss

def metric_hamming(true_label_list, predict_label_list, valid_label_list, threshold_list):
    loss_list = []
    
    total_true = []
    total_predict = []
    
    for index in range(0, len(true_label_list)):
        current_true_label = true_label_list[index]
        current_predict = predict_label_list[index]
        current_valid = valid_label_list[index]
        
        current_true_label = np.array(current_true_label)[current_valid]
                
        current_predict_thresholding = []
        for label_index, threshold in enumerate(threshold_list):
            current_result = 0
            if current_predict[label_index] > threshold:
                current_result = 1            
            current_predict_thresholding.append(current_result)    
                                
        current_predict_thresholding = np.array(current_predict_thresholding)[current_valid]      
          
        current_loss = calc_hamming_loss(current_true_label, current_predict_thresholding)
        
        loss_list.append(current_loss)
        
        total_true.extend(current_true_label)
        total_predict.extend(current_predict_thresholding)
    
    mean_loss = np.mean(loss_list)
    
    total_loss = calc_hamming_loss(total_true, total_predict)

    return mean_loss, total_loss

def metric_hamming_thresholding(true_label_list, predict_label_list, valid_label_list):
    loss_list = []
    for index in range(0, len(true_label_list)):
        current_true_label = true_label_list[index]
        current_predict = predict_label_list[index]
        current_valid = valid_label_list[index]
        
        current_true_label = np.array(current_true_label)[current_valid]
   
        current_predict_thresholding = np.array(current_predict)[current_valid]        
        current_loss = calc_hamming_loss(current_true_label, current_predict_thresholding)
        
        loss_list.append(current_loss)
    
    mean_loss = np.mean(loss_list)

    return mean_loss

def thresholdding(predict_label_list, threshold_list):
    result_thresholding = []
    for index in range(0, len(predict_label_list)):
        current_predict = predict_label_list[index]
        current_predict_thresholding = []
        for label_index, threshold in enumerate(threshold_list):
            current_result = 0
            if current_predict[label_index] > threshold:
                current_result = 1            
            current_predict_thresholding.append(current_result)    
        result_thresholding.append(current_predict_thresholding)         
       
    return result_thresholding

def calc_threshold(true_label_list, predict_label_list, valid_label_list, label_num=10, step=30):
    result_threshold = []
    current_threshold = [0.5] * label_num
    
    for index in range(label_num):    
        best_threshold = 0.5
        minimum_hamming = 1e10
        
        for threshold_value in np.linspace(0, 1, step): 
            current_threshold[index] = threshold_value
            current_hamming, _ = metric_hamming(true_label_list, predict_label_list, valid_label_list, current_threshold)
            
            if current_hamming < minimum_hamming:
                minimum_hamming = current_hamming
                best_threshold = threshold_value
        current_threshold[index] = best_threshold
        result_threshold.append(best_threshold)    
        
    return result_threshold
def get_evaluation_list(input_df, id_result=CSV_ID_RESULT):    
    current_gt = input_df[CSV_ID_GT_LABEL].values
    current_result = input_df[id_result].values
    current_valid = np.zeros((len(input_df), len(LABEL_ID)), dtype=bool)
    current_valid[:, :] = True
    current_valid[current_gt == 0.5] = False
    
    return current_gt, current_result, current_valid

def matching_data_from_ref(input_reference_df, input_target_df):
    reference_data_num = len(input_reference_df)
    target_data_num = len(input_target_df)

    #### matching data ####
    if reference_data_num != target_data_num:     
        small_df = input_reference_df
        large_df = input_target_df
        
        if target_data_num < reference_data_num:
            small_df = input_target_df
            large_df = input_reference_df
        
        small_image_list = np.array(small_df[CSV_ID_IMAGE].values)
        large_image_list = np.array(large_df[CSV_ID_IMAGE].values)
        
        index_no_small = []
        for current_index, current_image in enumerate(large_image_list):
            index = np.where(small_image_list == current_image)[0]
            if len(index) <= 0:
                index_no_small.append(current_index)
        
        ##### large에서 remove
        large_df.drop(index_no_small, axis=0, inplace=True)
        return large_df
    else:
        return input_target_df

original_df = pd.read_csv(ORIGINAL_CSV_PATH)
reference_df = pd.read_csv(REFERENCE_CSV_PATH)
target_1_df = pd.read_csv(TARGET_1_CSV_PATH)
target_2_df = pd.read_csv(TARGET_2_CSV_PATH)

reference_df = matching_data_from_ref(original_df, reference_df)
target_1_df = matching_data_from_ref(original_df, target_1_df)
target_2_df = matching_data_from_ref(original_df, target_2_df)

reference_gt, reference_result, reference_valid = get_evaluation_list(reference_df)
target_gt, target_1_result, target_valid = get_evaluation_list(target_1_df, CSV_ID_RESULT_TARGET_1)
target_gt, target_2_result, target_valid = get_evaluation_list(target_2_df, CSV_ID_RESULT_TARGET_2)

result_select = np.zeros(reference_result.shape, reference_result.dtype)
result_select[:, SELECT_NUM_TARGET_1] = target_1_result
result_select[:, SELECT_NUM_TARGET_2] = target_2_result
ensemble_result = 0.5*reference_result + 0.5*result_select

#save result to csv (Select: Ensemble)
image_list = original_df[CSV_ID_IMAGE].values
# select_df = pd.DataFrame({CSV_ID_IMAGE: image_list})
# ensemble_df = pd.DataFrame({CSV_ID_IMAGE: image_list})
# select_df[CSV_ID_GT_LABEL] = reference_gt
# ensemble_df[CSV_ID_GT_LABEL] = reference_gt
# select_df[CSV_ID_RESULT] = result_select
# ensemble_df[CSV_ID_RESULT] = result_select

# select_df.to_csv(OUTPUT_CSV_PATH_SELECT, index=False)
# ensemble_df.to_csv(OUTPUT_CSV_PATH_ENSEMBLE, index=False)

# print(" ######## result for reference ########## ")
# select_threshold = calc_threshold(reference_gt, reference_result, reference_valid)
# select_hamming_loss, _ = metric_hamming(reference_gt, reference_result, reference_valid, select_threshold)
# print("refference hamming loss = ", select_hamming_loss)
# print(select_threshold)

# roc_auc_list, sensitivity_list, specificity_list, threshold_list= metric_each_label(reference_gt, reference_result, reference_valid)
# mean_auc = np.mean(roc_auc_list)
# print("mean ROC-AUC: ", mean_auc)
# print(roc_auc_list)

# print(" ######## result for select ########## ")
# select_threshold = calc_threshold(reference_gt, result_select, reference_valid)
# select_hamming_loss, _ = metric_hamming(reference_gt, result_select, reference_valid, select_threshold)
# print("Select hamming loss = ", select_hamming_loss)
# print(select_threshold)

# roc_auc_list, sensitivity_list, specificity_list, threshold_list= metric_each_label(reference_gt, result_select, reference_valid)
# mean_auc = np.mean(roc_auc_list)
# print("mean ROC-AUC: ", mean_auc)
# print(roc_auc_list)

print(" ######## result for Ensemble (step 30) ########## ")
ensemble_threshold = calc_threshold(reference_gt, ensemble_result, reference_valid)
ensemble_hamming_loss, _ = metric_hamming(reference_gt, ensemble_result, reference_valid, ensemble_threshold)
print("Ensemble hamming loss = ", ensemble_hamming_loss)
print(ensemble_threshold)

roc_auc_list, sensitivity_list, specificity_list, threshold_list= metric_each_label(reference_gt, ensemble_result, reference_valid)
mean_auc = np.mean(roc_auc_list)
print("mean ROC-AUC: ", mean_auc)
print(roc_auc_list)

print(" ######## result for Ensemble (step 100) ########## ")
ensemble_threshold = calc_threshold(reference_gt, ensemble_result, reference_valid, step=100)
ensemble_hamming_loss, _ = metric_hamming(reference_gt, ensemble_result, reference_valid, ensemble_threshold)
print("Ensemble hamming loss = ", ensemble_hamming_loss)
print(ensemble_threshold)

roc_auc_list, sensitivity_list, specificity_list, threshold_list= metric_each_label(reference_gt, ensemble_result, reference_valid)
mean_auc = np.mean(roc_auc_list)
print("mean ROC-AUC: ", mean_auc)
print(roc_auc_list)   
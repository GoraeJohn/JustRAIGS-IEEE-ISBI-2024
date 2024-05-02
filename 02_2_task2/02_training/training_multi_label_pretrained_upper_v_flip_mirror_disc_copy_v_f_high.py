import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import cv2
import gc
import random
import timm
import logging
import albumentations as A
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc

IS_NPY = True
RAW_PATH = "/root/home/challenge/justraigs/multi_label"
INPUT_PATH = "/root/home/challenge/justraigs/train_data_preprocess_npy"
INDEX_PATH = os.path.join(RAW_PATH, 'index_list')

MODEL_PATH = os.path.join(RAW_PATH, "convnext_small_5fold_pretrained_v_flip_mirror_disc_copy_flip_high")
CSV = os.path.join(RAW_PATH, "label_info", "JustRAIGS_Train_labels_multi_from_G3.csv")

INDEX_PREFIX = 'multi_labeled'
POST_FIX = 'upper_v_flip_mirror_disc_copy_flip' 

CSV = CSV.replace(".csv", f"_{POST_FIX}.csv")
INDEX_PREFIX_TRAIN = f'{INDEX_PREFIX}_train_index_{POST_FIX}'
INDEX_PREFIX_VALID = f'{INDEX_PREFIX}_valid_index'

# PRETRAINED_PATH = ''
PRETRAINED_PATH = '/root/home/challenge/justraigs/multi_label/pretrained/binary_trained_240319'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0" #GPU ID

#LABEL_ID = ['ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']
LABEL_ID = ['ANRS', 'ANRI',  'NVT', 'LD', 'LC'] 
CSV_ID_PREFIX_GT = 'gt_'
CSV_ID_PREFIX_GT1 = 'G1 '
CSV_ID_PREFIX_GT2 = 'G2 '
CSV_ID_PREFIX_GT3 = 'G3 '

CSV_ID_IMAGE_NAME = 'Eye ID'
CSV_ID_EXT = 'image_ext'

class CFG:    
    # model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    model_name = "convnext_small.in12k_ft_in1k"
    #model_name = "convnext_base.fb_in22k_ft_in1k"
    img_size = 512
    max_epoch = 21
    batch_size = 32
    fold_num = 5
    out_dim = len(LABEL_ID)
    lr = 1.0e-05 #originaly lr =
    step_size=2
    gamma=1.0e-02
    seed = 1086
    deterministic = True
    enable_amp = True
    es_patience =  5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
df_val = pd.read_csv(CSV)
df_val_gt = df_val['gt_binary']

result_csv = os.path.join(MODEL_PATH, f'val_{MODEL_PATH.split("/")[-1]}.csv')
threshold_npy = os.path.join(MODEL_PATH, f'threshold_{MODEL_PATH.split("/")[-1]}.npy')
CSV_ID_RESULT = []

CSV_ID_GT_LABEL = []
CSV_ID_G1_LABEL = []
CSV_ID_G2_LABEL = []
CSV_ID_G3_LABEL = []
for current_label in LABEL_ID:
    CSV_ID_GT_LABEL.append(CSV_ID_PREFIX_GT + current_label)
    CSV_ID_G1_LABEL.append(CSV_ID_PREFIX_GT1 + current_label)
    CSV_ID_G2_LABEL.append(CSV_ID_PREFIX_GT2 + current_label)
    CSV_ID_G3_LABEL.append(CSV_ID_PREFIX_GT3 + current_label)
    CSV_ID_RESULT.append("result_" + current_label)    

train_transform =  A.Compose([        
        A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=(0.0, 0.2), rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),                        
        A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
    ])

valid_transform = A.Compose([        
        A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),        
    ])
        
#Model
class Timm_model(nn.Module):
    def __init__(self, model_name, out_dim=1,pretrained=True,  fc_num=1024, drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        self.model_name = model_name      
        print(timm.models)      
        #self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim, drop_rate=drop_rate, drop_path_rate=drop_path_rate)                            
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=fc_num, drop_rate=drop_rate, drop_path_rate=drop_path_rate)                            
        #To out_dim...
        self.fc = nn.Linear(in_features=fc_num, out_features=out_dim)
        
        self.input_shape = self.model.default_cfg['input_size']
      
    def get_input_shape(self):
        return self.input_shape

    def get_model_name(self):
        return self.model_name
    
    def forward(self, x):
        features = self.model(x)        
        output = self.fc(features)
        
        return output
    
    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)
        torch.save(self.fc.state_dict(), file_name.replace(".pth", "_fc.pth"))
    
    def load_model(self, file_name, device, is_read_fc=True):
        self.model.load_state_dict(torch.load(file_name, map_location=device))
        if is_read_fc:
            self.fc.load_state_dict(torch.load(file_name.replace(".pth", "_fc.pth")), map_location=device)
        
#Data generator
class CFPGenerator(Dataset):    
    def __init__(self, device, input_df, image_path, transform=None, mode='test', gt_list = [], gt_1_list = [], gt_2_list = [], gt_3_list = []):        
        self.image_path = image_path          
        self.transform = transform        
        self.device = device
        self.input_df = input_df
        self.mode = mode
        self.gt_list = gt_list
        self.gt_1_list = gt_1_list
        self.gt_2_list = gt_2_list
        self.gt_3_list = gt_3_list
        
    def __len__(self):
        return len(self.input_df)
    
    def __getitem__(self, i):   
        row = self.input_df.iloc[i]
    
        current_image_name = row[CSV_ID_IMAGE_NAME]
        
        if IS_NPY:
            current_ext_name = 'npy'
            current_image_path = os.path.join(self.image_path, f'{current_image_name}.{current_ext_name}')
            
            current_image = np.load(current_image_path)
        else:
            current_ext_name = row[CSV_ID_EXT]
            current_image_path = os.path.join(self.image_path, f'{current_image_name}.{current_ext_name}')
            
            current_image = cv2.imread(current_image_path, cv2.IMREAD_COLOR)
            
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=current_image)
            current_image = res['image']
        
        image = np.transpose(current_image, (2, 0, 1))
        image = torch.tensor(image)
        image_device = image.to(self.device)
        image_device = image_device.float() / 255.0   
        
        current_label = np.zeros(len(self.gt_list), dtype=np.float32)
        agreed_feature = [False] * len(self.gt_list)
        label_eval = [0] * len(self.gt_list)
        
        if self.mode != 'test':
            
            current_label[:] = row[self.gt_list]
            
            gt_1_label = row[self.gt_1_list]
            gt_2_label = row[self.gt_2_list]
            gt_3_label = row[self.gt_3_list] #Exists or Not
            if gt_3_label.hasnans == False:
                agreed_feature = [True] * len(self.gt_3_list)
                label_eval = gt_3_label.values
            else:                
                agreed_feature = np.equal(gt_1_label.values, gt_2_label.values)
                label_eval = gt_2_label.values                                                                   
            
        label = torch.tensor(current_label).float()
        label_device = label.to(self.device)
        
        agreed_feature = torch.tensor(agreed_feature)
        label_eval = torch.tensor(label_eval.tolist())
        
        return image_device, label_device, agreed_feature, label_eval

def seed_everything(seed: int = 42, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = deterministic  # type: ignore
    
seed_everything(CFG.seed, CFG.deterministic)

def train_func(train_loader, model, criterion, optimizer):
    model.train()

    if CFG.enable_amp:
        scaler = torch.cuda.amp.GradScaler()
    losses = []
    for batch_idx, (images, targets, agreed_feature, label_eval) in enumerate(train_loader):        
        if CFG.enable_amp:
            with torch.cuda.amp.autocast():
                logits = model(images).squeeze(-1)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            logits = model(images).squeeze(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item())

    scheduler.step()
    loss_train = np.mean(losses)
    
    return loss_train

def inference_func(valid_loader, model, criterion):
    model.eval()

    predict_list = []
    targets_list = []
    losses = []
    agreed_feature_list = []
    label_eval_list = []

    with torch.no_grad():
        for batch_idx, (images, targets, agreed_feature, label_eval) in enumerate(valid_loader):            
            logits = model(images).squeeze(-1)
            loss = criterion(logits, targets)            
            losses.append(loss.cpu().tolist())
                        
            probs = logits.sigmoid()
            
            predict_list.extend(probs.cpu().tolist())            
            targets_list.extend(targets.detach().cpu().tolist())
            agreed_feature_list.extend(agreed_feature.tolist())
            label_eval_list.extend(label_eval.tolist())
                        
    loss_val = np.mean(losses)
    
    return targets_list, predict_list, loss_val, agreed_feature_list, label_eval_list

#Val Function: auc of each label, total = Hamming distance
#Val Function for challenge, only hamming distance of each labelers...
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
    
    mean_loss = np.mean(loss_list)

    return mean_loss

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

def metric_each_label(true_label_list, predict_label_list, valid_label_list):
    target_list = [[] for _ in range(CFG.out_dim)]
    predict_list = [[] for _ in range(CFG.out_dim)]
    for index in range(len(true_label_list)):        
        for index_label in range(CFG.out_dim):
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
    
    for index_label in range(CFG.out_dim):
        current_roc_auc, sensitivity_best, specificity_best, threshold_best = metric_auc_sensitivity(target_list[index_label], predict_list[index_label])
        roc_auc_list.append(current_roc_auc)
        sensitivity_list.append(sensitivity_best)
        specificity_list.append(specificity_best)
        threshold_list.append(threshold_best)
        
    return roc_auc_list, sensitivity_list, specificity_list, threshold_list
    
                                
criterion = torch.nn.MultiLabelSoftMarginLoss()

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

#SET LOGGING
timestr = time.strftime("%Y%m%d-%H%M%S")
log = logging.getLogger()
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)

file_handler = logging.FileHandler(f'{MODEL_PATH}/log_train_{timestr}.txt')
file_handler.setFormatter(formatter)
log.addHandler(file_handler)
start_time_ = datetime.now() + timedelta(hours=9)
log.info(f"Image size: {CFG.img_size} Batch size: {CFG.batch_size} lr: {CFG.lr}")
log.info(f"Start time : {start_time_.strftime('%Y-%m-%d %H:%M:%S')}")

#n-FOLD validation
val_predict = np.zeros((len(df_val), len(LABEL_ID)), dtype=np.float32)
val_label = np.zeros((len(df_val), len(LABEL_ID)), dtype=np.int32)
val_agree = np.zeros((len(df_val), len(LABEL_ID)), dtype=bool)

for i in range(0, CFG.fold_num):
    
    log.info('#'*25)
    log.info(f'### Fold {i}')
    
    model = Timm_model(CFG.model_name, CFG.out_dim, True)
    model.to(CFG.device)
    
    if PRETRAINED_PATH != '':
        current_model_path = os.path.join(PRETRAINED_PATH, f'model_sen_{i}.pth')
        model.load_model(current_model_path, CFG.device, is_read_fc=False)        
        
    train_index_file_fath = os.path.join(INDEX_PATH, f'{INDEX_PREFIX_TRAIN}_{i}.npy')
    valid_index_file_fath = os.path.join(INDEX_PATH, f'{INDEX_PREFIX_VALID}_{i}.npy')
    
    train_index = np.load(train_index_file_fath)
    valid_index = np.load(valid_index_file_fath)
    
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CFG.step_size, gamma=CFG.gamma)
        
    train_dataset = CFPGenerator(CFG.device, df_val.iloc[train_index], INPUT_PATH, transform=train_transform, mode='train', gt_list=CSV_ID_GT_LABEL, gt_1_list=CSV_ID_G1_LABEL, gt_2_list=CSV_ID_G2_LABEL, gt_3_list=CSV_ID_G3_LABEL)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=0)
    valid_dataset = CFPGenerator(CFG.device, df_val.iloc[valid_index], INPUT_PATH, transform=valid_transform, mode='valid', gt_list=CSV_ID_GT_LABEL, gt_1_list=CSV_ID_G1_LABEL, gt_2_list=CSV_ID_G2_LABEL, gt_3_list=CSV_ID_G3_LABEL)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=0)

    log.info(f'### train size {len(train_index)}, valid size {len(valid_index)}')
    print("Train label count")
    for id in CSV_ID_GT_LABEL:
        count_0 = df_val.iloc[train_index][id].values.tolist().count(0)
        count_0_5 = df_val.iloc[train_index][id].values.tolist().count(0.5)
        count_1 = df_val.iloc[train_index][id].values.tolist().count(1)
        print(f'{id} Train: 0 = {count_0}, 0.5 = {count_0_5}, 1 = {count_1}')
        count_0 = df_val.iloc[valid_index][id].values.tolist().count(0)
        count_0_5 = df_val.iloc[valid_index][id].values.tolist().count(0.5)
        count_1 = df_val.iloc[valid_index][id].values.tolist().count(1)
        print(f'{id} Valid: 0 = {count_0}, 0.5 = {count_0_5}, 1 = {count_1}')
    log.info('#'*25)    
    best_val_loss = 1.0e+09
    best_epoch = 0
    best_hamming = 1.0e+09    
    best_auc = 1.0e-09
    
    for epoch in range(1, CFG.max_epoch + 1):
        train_loss = train_func(train_loader, model, criterion, optimizer)
        targets, predict_sigmoid, val_loss, agree_feature, label_feature = inference_func(valid_loader, model, criterion)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_model(os.path.join(MODEL_PATH, f'model_fold_{i}.pth'))            
            
        threshold_list = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]        
        roc_auc_list, sensitivity_list, specificity_list, threshold_list= metric_each_label(label_feature, predict_sigmoid, agree_feature)
        mean_auc = np.mean(roc_auc_list)
        
        log.info(f"[epoch {epoch}] train loss: {train_loss: .6f}, val loss: {val_loss: .6f}, mean_auc: {mean_auc: .6f}")
        each_auc_str = "Each auc = "
        for current_auc in roc_auc_list:
            each_auc_str += f'{current_auc: .6f}, '
        log.info(each_auc_str)
            
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_auc_threshold = threshold_list
            best_epoch = epoch
            model.save_model(os.path.join(MODEL_PATH, f'model_auc_fold_{i}.pth'))
            val_predict[valid_index, :] = predict_sigmoid
            val_label[valid_index, :] = label_feature
            val_agree[valid_index, :] = agree_feature
            
        if epoch - best_epoch > CFG.es_patience:
            log.info("Early Stopping!")
            break
            
    del scheduler
    del optimizer
    del model
    torch.cuda.empty_cache()
    gc.collect()

np.save(os.path.join(MODEL_PATH, 'best_auc_threshold.npy'), best_auc_threshold)

log.info('#'*25)    
log.info("Total Validation")
log.info('#'*25)    
predict_list = val_predict.tolist()
label_list = val_label.tolist()
agree_list = val_agree.tolist()

roc_auc_list, sensitivity_list, specificity_list, threshold_list= metric_each_label(label_list, predict_list, agree_list)
hamming_loss = metric_hamming(label_list, predict_list, agree_list, threshold_list)    
mean_auc = np.mean(roc_auc_list)

log.info(f"val loss: {val_loss: .6f}, mean_auc: {mean_auc: .6f}, hamming_loss: {hamming_loss: .6f}")
each_auc_str = "Each auc = "
for current_auc in roc_auc_list:
    each_auc_str += f'{current_auc: .6f}, '
log.info(each_auc_str)

for index in range(len(LABEL_ID)):
    df_val[CSV_ID_RESULT[index]] = val_predict[:, index]
    
df_val.to_csv(result_csv, index=False)
np.save(threshold_npy, threshold_list)

log.info("finish")
###
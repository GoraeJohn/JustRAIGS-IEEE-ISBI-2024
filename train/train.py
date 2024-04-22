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
from sklearn.metrics import roc_curve, auc

RAW_PATH = "/root/challenge/justraigs"
INPUT_PATH = os.path.join(RAW_PATH, "images/merge_preprocess_512")
MODEL_PATH = os.path.join(RAW_PATH, "models/convnext_small_5fold_upsampling_train_dropset_valid_pretrained_batch8_240419")
INDEX_PATH = os.path.join("/root/challenge/justraigs/multi_label", 'index_list')
CSV = os.path.join(RAW_PATH, "csv/JustRAIGS_Train_labels_segment_from_G3_revised_pair_removed_final_upsampling.csv")
VALID_CSV = os.path.join(RAW_PATH, "csv/JustRAIGS_Train_labels_segment_from_G3_revised_pair_removed_final.csv")

PRETRAINED_PATH = '/root/challenge/justraigs/pre-trained/convnext_small_5fold_upsampling_train_dropset_valid_240402'

TRAIN_INDEX_PREFIX = 'smooth_labeled_drop_dataset_upsampling'
VALID_INDEX_PREFIX = 'multi_labeled_with_normal'
    
INDEX_PREFIX_TRAIN = f'{TRAIN_INDEX_PREFIX}_train_index'
INDEX_PREFIX_VALID = f'{VALID_INDEX_PREFIX}_valid_index'

class CFG:
    model_name = "convnext_small.in12k_ft_in1k"
    img_size = 512
    max_epoch = 21
    batch_size = 8
    fold_num = 5
    out_dim = 1
    lr = 1.0e-05
    lr_lion = 1.0e-04
    step_size=2
    gamma=1.0e-02
    seed = 1086
    deterministic = True
    enable_amp = True
    es_patience =  5
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
df_val = pd.read_csv(CSV)
df_val_ID = df_val['Eye ID']
df_val_gt = df_val['gt_binary']
df_val_gt_smooth = df_val['gt_binary_smooth']

df_valid = pd.read_csv(VALID_CSV)
df_valid_ID = df_valid['Eye ID']
df_valid_gt = df_valid['gt_binary']

train_transform =  A.Compose([                                   
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),        
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=(0.0, 0.2), rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),                        
        A.Resize(p=1.0, height=512, width=512)
    ])
        
#Model
class Timm_model(nn.Module):
    def __init__(self, model_name, out_dim=1, pretrained=True, fc_num=1024, drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        self.model_name = model_name      
        print(timm.models)      
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
            self.fc.load_state_dict(torch.load(file_name.replace(".pth", "_fc.pth"), map_location=device))

class SmoothCrossEntropy(nn.Module):
    def __init__(self):
        super(SmoothCrossEntropy(), self).__init()
        
    def forward(self, x, target, smoothing=0.0):
        confidence = 1. - smoothing
        logprobs = nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

#Data generator
class CFPGenerator(Dataset):
    def __init__(self, device, image_data, gt_data, transform=None, aug_channel=False, mode='train', height=128, width=256, num_label=2):
        self.image_data = image_data    
        self.gt_data = gt_data    
        self.transform = transform
        self.mode = mode
        self.height = height
        self.width = width
        self.num_label = num_label
        self.device = device
        self.aug_channel = aug_channel
        
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, i):   
        current_label = self.gt_data[i]
        image_name = self.image_data[i]
        current_image_name = image_name + '.BMP'
        current_image = cv2.imread(os.path.join(INPUT_PATH, current_image_name))
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None :
            transform = self.transform(image=current_image)
            current_image = transform['image']
        
        image = np.transpose(current_image, (2, 0, 1))
        image = torch.tensor(image)
        image_device = image.to(self.device)
        image_device = image_device.float() / 255.0
        
        label = torch.tensor(current_label).float()
        label_device = label.to(self.device)

        return image_device, label_device

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
    for batch_idx, (images, targets) in enumerate(train_loader):
        targets = targets.float()

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

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(valid_loader):
            targets = targets.float()
            logits = model(images).squeeze(-1)
            loss = criterion(logits, targets)
            losses.append(loss.cpu().tolist())
            probs = logits.sigmoid()
            predict_list.extend(probs.cpu().tolist())
            targets_list.extend(targets.detach().cpu().tolist())

    loss_val = np.mean(losses)
    
    return targets_list, predict_list, loss_val

#Val Function
def validation_func_task1(targets, predicts):
    
    desired_specificity = 0.95
    fpr, tpr, thresholds = roc_curve(targets, predicts)
    roc_auc = auc(fpr, tpr)
    idx = np.argmax(fpr >= (1 - desired_specificity))
    threshold_at_desired_specificity = thresholds[idx]
    sensitivity_at_desired_specificity = tpr[idx]
        
    return roc_auc, sensitivity_at_desired_specificity, threshold_at_desired_specificity

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
log.info(f"{train_transform}")
log.info(f"{INPUT_PATH}")
log.info(f"Image size: {CFG.img_size} Batch size: {CFG.batch_size} lr: {CFG.lr}")
log.info(f"Start time : {start_time_.strftime('%Y-%m-%d %H:%M:%S')}")

#n-FOLD validation
from sklearn.model_selection import StratifiedKFold

all_predictions_softmax = []
all_prediction_sigmoid = []
all_target = []

skf = StratifiedKFold(n_splits=CFG.fold_num)
for i in range(0, CFG.fold_num):
    
    log.info('#'*25)
    log.info(f'### Fold {i}')
    
    model = Timm_model(CFG.model_name, CFG.out_dim, True)
    model.to(CFG.device)

    train_index_file_path = os.path.join(INDEX_PATH, f'{INDEX_PREFIX_TRAIN}_{i}.npy')
    valid_index_file_fath = os.path.join(INDEX_PATH, f'{INDEX_PREFIX_VALID}_{i}.npy')
    train_index = np.load(train_index_file_path)
    valid_index = np.load(valid_index_file_fath)

    criterion = torch.nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CFG.step_size, gamma=CFG.gamma)
        
    train_dataset = CFPGenerator(CFG.device, df_val_ID.iloc[train_index].values, df_val_gt_smooth.iloc[train_index].values, transform=train_transform, aug_channel=False, mode='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=0)
    valid_dataset = CFPGenerator(CFG.device, df_valid_ID.iloc[valid_index].values, df_valid_gt.iloc[valid_index].values, transform=None, aug_channel=False, mode='valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=0)

    log.info(f'### train size {len(train_index)}, valid size {len(valid_index)}')
    log.info('#'*25)    
    best_val_loss = 1.0e+09
    best_epoch = 0
    train_loss = 0
    best_auc = 1.0e-09
    best_sigmoid_norm = 1.0e-09
    best_loss = 1.0e+09
    best_sen = 1.0e-09
    best_thre = 1.0e-09
    
    for epoch in range(1, CFG.max_epoch + 1):
        train_loss = train_func(train_loader, model, criterion, optimizer)
        targets, predicts, val_loss = inference_func(valid_loader, model, criterion)
        current_auc, current_sen, current_thre = validation_func_task1(targets, predicts)
                                 
        if current_auc > best_auc:
            best_auc = current_auc      
            Timm_model.save_model(model, os.path.join(MODEL_PATH, f'model_auc_{i}.pth'))                  
            
        if current_sen > best_sen:
            best_sen = current_sen
            best_thre = current_thre
            best_epoch = epoch
            Timm_model.save_model(model, os.path.join(MODEL_PATH, f'model_sen_{i}.pth'))
            
        log.info(
            f"[epoch {epoch}] train loss: {train_loss: .6f}, val loss: {val_loss: .6f}, val sensitivity: {best_sen: .6f}, val threshold: {best_thre: .6f}, AUC: {best_auc: .6f}")
        
        if epoch - best_epoch > CFG.es_patience:
            log.info("Early Stopping!")
            break
            
    del scheduler
    del optimizer
    del model
    torch.cuda.empty_cache()
    gc.collect()

log.info("finish")
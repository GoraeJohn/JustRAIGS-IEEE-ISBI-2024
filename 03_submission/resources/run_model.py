## Pytorch model.. Inference/Sigmoid 

##ModelClass
import torch
import torch.nn as nn
import timm
import numpy as np

MODEL_NAME = "convnext_small.in12k_ft_in1k"

class Timm_model_1(nn.Module):
    def __init__(self, model_name, out_dim=1, pretrained=False, drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        self.model_name = model_name      
        print(timm.models)      
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim, drop_rate=drop_rate, drop_path_rate=drop_path_rate)                            
        self.input_shape = self.model.default_cfg['input_size']
      
    def get_input_shape(self):
        return self.input_shape

    def get_model_name(self):
        return self.model_name
    
    def forward(self, x):
        output = self.model(x)        
        
        return output
        
class Timm_model_2(nn.Module):
    def __init__(self, model_name, out_dim=1,pretrained=False,  fc_num=1024, drop_rate=0.0, drop_path_rate=0.0):
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
    
    def load_model(self, file_name, device):
        self.model.load_state_dict(torch.load(file_name, map_location=device))
        self.fc.load_state_dict(torch.load(file_name.replace(".pth", "_fc.pth"), map_location=device))
        

class RunTorch:
    def __init__(self, model_file_path_list = [], out_num=1, model_version=2, is_sigmoid=True, is_mean=True, model_name=MODEL_NAME):
        self.__is_sigmoid = is_sigmoid        
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__is_mean = is_mean
        self.__model_list = []
        for current_file in model_file_path_list:
            if model_version==1:                
                current_model = Timm_model_1(model_name, out_num)
                current_model.to(self.__device)
                current_model.load_state_dict(torch.load(current_file, map_location=self.__device))
            else:
                current_model = Timm_model_2(model_name, out_num)
                current_model.to(self.__device)
                current_model.load_model(current_file, self.__device)            
            
            self.__model_list.append(current_model)
        
    def run(self, current_image):
        ##current image -> to device
        current_image_tensor = torch.tensor(current_image).to(self.__device)
        
        ##result
        result_list = []
        for current_model in self.__model_list:
            current_model.eval()
            logit = current_model(current_image_tensor).squeeze(-1)
            if self.__is_sigmoid:
                probs = logit.sigmoid()
            else:
                probs = logit.softmax(dim=1)[:, 1]
            result_list.append(probs.cpu().tolist())
            
        result_np = np.array(result_list)
        
        if self.__is_mean:
            result_value = np.mean(result_np, axis=0) #shape [batch_size, out_num]
        else:
            result_value = np.max(result_np, axis=0)
                
        return result_value.tolist()
    
    def run_single_batch(self, current_image):
        result_value = self.run(current_image)
        
        return result_value[0]
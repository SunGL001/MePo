import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import random
import timm
from timm.utils import accuracy
import numpy as np
import copy
from utils.data_manager import DataManager
import argparse
import wandb



parser = argparse.ArgumentParser()
parser.add_argument('-gpu_ids', default='0,1', type=str, help='gpu ids for training')
parser.add_argument('-inner_lr', type=float, default=0.0001, help='kl divergence temperature')
parser.add_argument('-num_inner_steps', type=int, default=4, help='number of inner steps')
parser.add_argument('-meta_lr', type=float, default=0.01, help='meta learning rate')
parser.add_argument('-batch_size', type=int, default=256, help='batch size for dataloader')
parser.add_argument('-num_tasks', type=int, default=10, help='number of tasks for continual learning')
parser.add_argument('-samples', type=int, default=10, help='number of tasks for continual learning')
parser.add_argument('-epochs', type=int, default=50, help='epochs')
parser.add_argument('-model', default='sup', choices=['sup', 'dino','ibot', 'sup1k'], type=str, help='pre-trained model')
parser.add_argument('-wandb_entity', default='none', type=str, help='downsample size of teacher model')
parser.add_argument('-wandb_project', default='meta', type=str, help='downsample size of student model')
parser.add_argument('task_inc', action='store_true', default=False, help='use task_mask fortask incremental learning')
_arg = parser.parse_args()


# 配置参数
os.environ['CUDA_VISIBLE_DEVICES'] = _arg.gpu_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args=dict()
args['sample_number'] = _arg.samples
args['init_cls'] = 100
args['increment'] = 100
args['classes_per_task'] = int(1000 / _arg.num_tasks)
args['num_tasks'] = _arg.num_tasks
args['num_meta_epochs'] = _arg.epochs
args['inner_lr'] = _arg.inner_lr
args['meta_lr'] = _arg.meta_lr
args['num_inner_steps'] = _arg.num_inner_steps
args['batch_size'] = _arg.batch_size
args['seed'] = 1
# args['classifier'] = ''

seed = args['seed']
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# Reptile元学习算法
class Reptile:
    def __init__(self, model, meta_lr):
        self.meta_model = model
        self.meta_lr = meta_lr
        
    def meta_update(self, initial_weights, updated_weights_list):
        total_delta = {k: torch.zeros_like(v) for k, v in initial_weights.items()}
        
        # 计算所有任务的权重变化均值
        for weights in updated_weights_list:
            for k in initial_weights:
                total_delta[k] += (weights[k] - initial_weights[k]) / len(updated_weights_list)
                
        # 应用元更新
        new_weights = copy.deepcopy(initial_weights)
        for k in initial_weights:
            new_weights[k] += self.meta_lr * total_delta[k]
        self.meta_model.load_state_dict(new_weights)

# 初始化模型
class metaViT(nn.Module):
    def __init__(self, arg):
        super(metaViT, self).__init__()
        if arg.model=='sup':
            self.vit = timm.create_model('vit_base_patch16_224_in21k', pretrained=False, checkpoint_path="[your path]/ViT-B_16.npz") # vit_base_patch16_224_dino, vit_base_patch16_224_sam
            # model_infos=torch.load("[your path]/vit_base_patch16_224_in21k.pth")
            # filtered_model_infos = {k: v for k, v in model_infos.items()if not k.startswith("fc.")}
            # load_result = self.vit.load_state_dict(filtered_model_infos, strict=False)            
            # print("Missing keys:", load_result.missing_keys)
            # print("Unexpected keys:", load_result.unexpected_keys)
            # successful_keys = set(filtered_model_infos.keys()) - set(load_result.unexpected_keys)
            # print("Successfully loaded keys:", successful_keys)
        
        elif arg.model=='sup1k':
            self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

        elif arg.model=='dino':
            self.vit = timm.create_model('vit_base_patch16_224_dino', pretrained=True)

        elif arg.model=='ibot':
            self.vit = timm.create_model('vit_base_patch16_224_in21k', pretrained=False, checkpoint_path="[your path]/ViT-B_16.npz") # vit_base_patch16_224_dino, vit_base_patch16_224_sam          
            state_dict = self.vit.state_dict()
            #s_ckpt = torch.load('/home/work/xiejingyi/SLCA_code/convs/checkpoint.pth', map_location='cpu')['teacher']
            s_ckpt = torch.load("[your path] /checkpoint.pth", map_location='cpu')['teacher']
            ckpt = {}
            for key, val in s_ckpt.items():
                new_key = key.replace('backbone.', '')
                ckpt[new_key] = val
            not_in_k = [k for k in ckpt.keys() if k not in state_dict.keys()]
            for k in not_in_k:
                del ckpt[k]
            state_dict.update(ckpt)
            load_result=self.vit.load_state_dict(state_dict)
         
            # print("Missing keys:", load_result.missing_keys)
            # print("Unexpected keys:", load_result.unexpected_keys)
            # successful_keys = set(state_dict.keys()) - set(load_result.unexpected_keys)
            # print("Successfully loaded keys:", successful_keys)

        print(self.vit.default_cfg)
        self.feature_dim = 768
        self.vit.head = nn.Identity()
        # create a new head (feature_dim to feature_dim*4 to classes)
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 1000)
        )

    def forward(self, x, updata_head=False):
        x=self.vit(x)
        x=self.head(x)
        return x

model = metaViT(arg=_arg)
# model = timm.create_model('vit_base_patch16_224', pretrained=True)

# 多GPU支持
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model.to(device)

reptile_model = Reptile(model, args['inner_lr'])

param_groups = [
    {"params": [], "lr": 0.0001},  # lr=0.0001
    {"params": [], "lr": 0.01}    # head=0.01
]

param_groups_inner = [
    {"params": [], "lr": 0.00002},  # lr=0.0001
    {"params": [], "lr": 0.01}    # head=0.01
]

for name, param in model.named_parameters():
    # param.requires_grad = True
    if "head" in name:
        param_groups[1]["params"].append(param)
        param_groups_inner[1]["params"].append(param)
    else: 
        param_groups[0]["params"].append(param)
        param_groups_inner[0]["params"].append(param)
optimizer = torch.optim.SGD(param_groups, momentum=0.9)
# optimizer_inner = torch.optim.SGD(param_groups_inner, momentum=0.9)

# optimizer = torch.optim.SGD(model.parameters(), lr=args['inner_lr'])
# optimizer_in = torch.optim.SGD(model.module.head.parameters(), lr=0.01)
# optimizer_out = torch.optim.SGD(model.module.vit.parameters(), lr=args['inner_lr'])
print(f"Trainable Param Count: {sum(param.numel() for param in model.parameters() if param.requires_grad)}")
save_path=f"/data/meta_vit2"
model_id=f"R100_noW_{_arg.model}_sample_{args['sample_number']}_lr_{args['inner_lr']}_numtask_{args['num_tasks']}_steps_{args['num_inner_steps']}"

if not os.path.exists(os.path.join(save_path, model_id)):
        os.makedirs(os.path.join(save_path, model_id))
torch.save(model.module.vit.state_dict(), os.path.join(save_path, model_id, f"meta_epoch_{0}.pth"))


wandb.init(dir='/data/meta_vit/wandb_meta',entity=_arg.wandb_entity, project=_arg.wandb_project, name=model_id, config=args)
wandb_url = wandb.run.get_url()
print(f"Wandb URL: {wandb_url}")


# 训练循环
# reptile = Reptile(model, args)
for meta_epoch in range(args['num_meta_epochs']):
    print(f"Meta Epoch [{meta_epoch+1}/{args['num_meta_epochs']}]")
    increment=args['increment']
    data_manager=DataManager(dataset_name='imagenet1000', shuffle=True, seed=np.random.randint(100), 
    init_cls=increment, increment=increment, args=args)
    # class_mask
    class_order = data_manager._class_order
    class_mask = tuple([class_order[i:i+increment] for i in range(0, len(class_order), increment)])
    # print(f"class_order: {class_order}")
    
    initial_weights = copy.deepcopy(model.state_dict())
    updated_weights_list = []
    # 克隆元模型
    # task_model = copy.deepcopy(model)
    # task_model.load_state_dict(initial_weights)

    # initial weight of head
 
    # 内循环：遍历所有任务
    for task_id in range(10):
        print(f"Task [{task_id+1}/{10}]")
        # 准备任务数据
        initial_weights = copy.deepcopy(model.state_dict())
        updated_weights_list = []

        model.train()

        class_order2 = class_mask[task_id] # 100 classes
        increment2 = int(len(class_order2) / args['num_tasks'])
        class_mask2 = tuple([class_order2[i:i+increment2] for i in range(0, len(class_order2), increment2)])

        test_dataset = data_manager.get_dataset(class_order2,source="train", mode="test")
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=8)

        mask_labels = []
        for _, _, labels in test_loader:
            mask_labels.append(labels.numpy())
        mask_labels = np.concatenate(mask_labels)
        mask_labels = np.unique(mask_labels)
        mask_labels= np.array(class_order)[mask_labels]

        for rep_step in range(args['num_inner_steps']):

            # for name, param in model.named_parameters():
            #     if "head" in name:  # 根据实际层名匹配
            #         if "weight" in name:
            #             nn.init.xavier_normal_(param)  # 随机初始化权重
            #             # print(f"Initialize {name} with xavier normal")
            #         elif "bias" in name:
            #             nn.init.constant_(param, 0.0)  # 初始化偏置为0
            #             # print(f"Initialize {name} with constant 0")

            for i in range(len(class_mask2)):
                train_dataset = data_manager.get_dataset(class_mask2[i],source="train", mode="train")
                task_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=8)

                # 任务内训练
                for step in range(4):
                    # print(f"Inner Step [{step+1}/{args['num_inner_steps']}]")
                    total = 0
                    correct = 0
                    for _, inputs, labels in task_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        labels=torch.tensor(class_order)[labels.cpu().numpy()].to(device)
                        # optimizer_inner.zero_grad()
                        optimizer.zero_grad()
                        outputs = model(inputs, updata_head=True)

                        mask = torch.zeros_like(outputs)
                        mask[:, mask_labels] = 1
                        outputs = outputs * mask

                        _, predicted = torch.max(outputs, 1)  
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        loss = F.cross_entropy(outputs, labels)
                        loss.backward()

                        # print(f"Task {task_id+1} Inner Step {step+1} Acc: {correct/total:.2f}")
                        wandb.log({"meta_train_loss": loss.item()})

                        # optimizer_inner.step()
                        optimizer.step()

                    wandb.log({"meta_train_acc": (correct/total)*100 })

            # stat_matrix = np.zeros((args['num_tasks'],1))  # 2 for Acc@1
            # acc_matrix = np.zeros((args['num_tasks'], args['num_tasks']))

            total = 0
            correct = 0
            for _, inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels=torch.tensor(class_order)[labels.cpu().numpy()].to(device)
                optimizer.zero_grad()
                outputs = model(inputs, updata_head=False)

                mask = torch.zeros_like(outputs)
                mask[:, mask_labels] = 1
                outputs = outputs * mask

                _, predicted = torch.max(outputs, 1)  
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # print(f"Acc: {correct/total:.2f}")

                loss_out = F.cross_entropy(outputs, labels)
                loss_out.backward()
                optimizer.step()
                # print(f"loss_out: {loss_out.item()}")
                wandb.log({"meta_test_loss": loss_out.item()})
            wandb.log({"meta_test_acc": (correct/total)*100 })
            updated_weights_list.append(copy.deepcopy(model.state_dict()))
        # 应用元更新
        reptile_model.meta_update(initial_weights, updated_weights_list)

    if not os.path.exists(os.path.join(save_path, model_id)):
        os.makedirs(os.path.join(save_path, model_id))
    if meta_epoch == 0 or (meta_epoch+1) % 1 == 0:
        torch.save(model.module.vit.state_dict(), os.path.join(save_path, model_id, f"meta_epoch_{meta_epoch+1}.pth"))
    # # 测试模型
                
wandb.finish()
print("Training completed!")




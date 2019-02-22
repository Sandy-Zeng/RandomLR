import os

dataset_names = ['CIFAR100']
# dataset_names = ['CIFAR10']
batch_size = [128]  # orig paper trained all networks with batch_size=128
epochs = 200
optimizers = ['SGD_Mon']
distribution_methods = ['Base']
lr_schedule_methods = ['post_random']
random_ranges = [4]
init_lrs = [1e-1]
mulTs = [1]
TB_log_path = 'TB_Log'
work_path_name = 'FinalResult/Ablation/B2'
resnet_depth = [20]
random_portions  = [0.4]
model = 'resnet'
rlaunch = 'rlaunch --cpu=2 --memory=4096 --gpu=1 --preemptible=no '
# rlaunch = 'CUDA_VISIBLE_DEVICES=4 '
data_aug = 1
# rlaunch = '' #With bash source
for dataset_name in dataset_names:
    for bs in batch_size:
        for optimizer in optimizers:
            for distribution_method in distribution_methods:
                for init_lr in init_lrs:
                    for depth in resnet_depth:
                        for lr_schedule_method in lr_schedule_methods:
                            for p in random_portions:
                                for m in mulTs:
                                    for r in random_ranges:
                                        cmd = rlaunch + 'python3 ../train/RLR_Exp.py %s %d %d %s %s %s %f %f %s %s %d %s %s %d %f %d' \
                                          % (dataset_name, bs, epochs, optimizer, lr_schedule_method,distribution_method,r,init_lr,TB_log_path,work_path_name,depth,model,'triangular2',m,p,data_aug)
                                        os.system(cmd)
import os
# rlaunch = 'rlaunch --cpu=2 --memory=4096 --gpu=1 --preemptible=no '
rlaunch = 'CUDA_VISIBLE_DEVICES=0 '
datasets = ['MNIST']
distribution_method = ['U']
random_range = 4
sgd = 'sgd'
init_lr = 1e-1
word_path = 'Final/AAAI_Exp/'

for dataset_name in datasets:
    for method in distribution_method:
        if dataset_name == 'MNIST':
            cmd = rlaunch + 'python3 ../train/mnist_train.py %s %d %s %s %.2f'% (method,random_range,sgd,word_path,init_lr)
            os.system(cmd)
        if dataset_name == 'IMDB':
            cmd = rlaunch + 'python3 ../train/imdb_lstm.py %s %d' % (method, random_range)
            os.system(cmd)
        if dataset_name == 'SVHN':
            cmd = rlaunch + '-- python3 ../train/svhn_train.py --data_set_path /home/ouyangzhihao/Backup/dataset/SVHN --distribution_method %s --random_range %d' % (method,random_range)
            os.system(cmd)


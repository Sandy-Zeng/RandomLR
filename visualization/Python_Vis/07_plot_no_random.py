import matplotlib.pyplot as plt
import pickle as pk

with open('weights_loss_all_no_random.pkl', 'rb') as f:
    weights_loss_all = pk.load(f)

with open('weights_loss_train_no_random.pkl', 'rb') as f:
    weights_loss_train = pk.load(f)


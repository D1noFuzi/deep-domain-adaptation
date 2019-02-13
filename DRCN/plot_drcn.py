import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('./model_m2mm/run_1_taso/stats.pkl', 'rb') as f:
    results = pickle.load(f)
    key_list = ['source_loss_train', 'source_acc_train', 'source_acc_test',
                 'target_loss_train', 'target_acc_train', 'target_acc_test']

    x_1 = [x[0] for x in results[key_list[2]]]
    y_1 = [x[1] for x in results[key_list[2]]]
    x_2 = [x[0] for x in results[key_list[-1]]]
    y_2 = [x[1] for x in results[key_list[-1]]]
    print(np.mean(y_1))
    print(np.mean(y_2))
fig, ax = plt.subplots(1, 1)
axes = plt.gca()
axes.set_ylim([0, 1])
axes.set_xlim([0, 100])
plt.plot(x_1, y_1)
plt.plot(x_2, y_2)
plt.show()

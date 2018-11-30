import pickle as pkl
import numpy as np
import os
from scipy import ndimage as sci


def create_mnistm(impath, txtpath):
    path = os.path.join("..", "datasets", "mnist-m", "mnist_m")
    x_ = []
    y_ = []
    with open(os.path.join(path, txtpath)) as f:
        for line in f:
            img_name, label = line.split()
            img_path = os.path.join(path, impath)
            img = sci.imread(os.path.join(img_path, img_name))
            x_.append(img)
            y_.append(label)
    x_ = np.asarray(x_, dtype=np.uint8)
    y_ = np.asarray(y_, dtype=np.uint8)
    return x_, y_


X_train, Y_train = create_mnistm("mnist_m_train", "mnist_m_train_labels.txt")
X_test, Y_test = create_mnistm("mnist_m_test", "mnist_m_test_labels.txt")
print(X_train.shape)
with open(os.path.join("..", "datasets", "mnist-m", "mnistm_data.pkl"), "wb") as f:
    pkl.dump({'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test}, f, pkl.HIGHEST_PROTOCOL)


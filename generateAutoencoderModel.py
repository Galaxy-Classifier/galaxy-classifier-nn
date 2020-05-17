import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from autoencoder import Autoencoder

width, height = 100, 100


def extract_data(path):
    data = []
    files = glob.glob(path)
    for myFile in files:
        image = cv2.imread(myFile)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im_resized = cv2.resize(gray_image, (width, height), interpolation=cv2.INTER_LINEAR)
        data.append(im_resized)

    return np.array(data)

espiral = extract_data("./data/Espiral/*.jpg")
elliptical = extract_data("./data/Elliptical/*.jpg")
lenticular = extract_data("./data/Lenticular/*.jpg")

x_test = np.concatenate([espiral])
y_test = np.concatenate(
    [np.full(len(espiral), 'espiral')])

x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.2, random_state=13)

x_train = x_train.reshape(x_train.shape[0], width, height, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], width, height, 1).astype('float32')
x_train /= 255
x_test /= 255

input_shape = (width, height, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

aux = Autoencoder()
size= 100
aux.train(x_train, x_test, 256, 1000)
decoded_imgs = aux.getDecodedImage(x_test)

plt.figure(figsize=(20, 4))
for j in range(10):
    # Original
    subplot = plt.subplot(2, 10, j + 1)
    plt.imshow(x_test[j].reshape(size, size))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)
    # Reconstruction
    subplot = plt.subplot(2, 10, j + 11)
    plt.imshow(decoded_imgs[j].reshape(size, size))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)
plt.show()
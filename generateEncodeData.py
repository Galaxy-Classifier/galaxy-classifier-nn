from tensorflow.keras.models import load_model
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
url_dataset = "./data/Lenticular/*.jpg"
url_model = "./autoencoders/autoencoder_lenticular.h5"
url_save = "./encode_data/Lenticular/"
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

model = load_model(url_model)
images = extract_data(url_dataset)
images = np.concatenate([images])
for num,img in enumerate(images,start=1):
  img = img.reshape(1, width, height, 1).astype('float32')
  img /= 255
  encode_img = model.predict(img)
  plt.imshow(encode_img.reshape(width,height))
  plt.axis('off')
  plt.gray()
  plt.savefig(url_save+"fig_{}.jpg".format(num),format='jpg',transparent=True,bbox_inches='tight',pad_inches = 0)
  print("fig_{}.jpg".format(num))

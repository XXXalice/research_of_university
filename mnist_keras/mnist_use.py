from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('./src/mnist_model.h5')

img_path = './src/4.png'
img = image.load_img(img_path, grayscale=True, target_size=(28, 28))
x = image.img_to_array(img).reshape(-1,784)
res = model.predict(x)

print(np.argmax(res)+1)
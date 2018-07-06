from param import base_param as bp
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
import matplotlib.pyplot as plt

#全画像を1/255でスケーリング
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     directory=bp.GENARATE_FOLDER[0],
#     target_size=(50, 50),
#     batch_size=10,
#     class_mode='binary'
# )
#
# validation_generator = test_datagen.flow_from_directory(
#     directory=bp.GENARATE_FOLDER[1],
#     target_size=(50, 50),
#     batch_size=20,
#     class_mode='binary'
# )

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

fnames = [os.path.join(bp.GENARATE_FOLDER[0], fname) for fname in os.listdir(bp.GENARATE_FOLDER[0])]
fnames.sort()

img_path = fnames[0]


img = image.load_img(img_path, target_size=(100, 100))

x = image.img_to_array(img)

x = x.reshape((1, ) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break


plt.show()
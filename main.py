from keras.models import Sequential
from keras.layers import Convolution2D
from keras.callbacks import Callback
from PIL import Image
import numpy as np
from scipy.ndimage import filters

class LossHistory(Callback):
    def __init__(self):
        Callback.__init__(self)
        self.losses = []
    def on_train_begin(self, logs=None):
        pass
    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))

lena = np.array(Image.open("lena.jpg").convert("L"))
lena_sobel = np.zeros(lena.shape)

#sobel算子
sobel = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
])

#计算卷积：用sobel算子滤波，结果保存在lena_sobel中
filters.convolve(input = lena,output = lena_sobel,weights = sobel,cval = 1.0)

lena_tmp = np.uint8((lena_sobel - lena_sobel.min()) * 255 / (lena_sobel.max() - lena_sobel.min()))
Image.fromarray(lena_tmp).save("lena_sobel.png")

X = lena.reshape((1,1) + lena.shape)
Y = lena_sobel.reshape((1,1) + lena_sobel.shape)

model = Sequential()
model.add(Convolution2D(nb_filter = 1,nb_row = 3,nb_col = 3,dim_ordering = "th",input_shape=X.shape[1:],border_mode="same",bias = False,init = "uniform"))
model.compile(loss = "mse",optimizer = "rmsprop",metrics = ["accuracy"])
history = LossHistory()

for i in range(0,10):
    lena_tmp = model.predict(X).reshape(lena.shape)
    lena_tmp = np.uint8((lena_tmp - lena_tmp.min()) * 255 / (lena_tmp.max() - lena_tmp.min()))
    Image.fromarray(lena_tmp).save("lena_sobel_stage_{:d}.png".format(i))
    print("lena_sobel_stage_{:d}.png saved".format(i))

    model.fit(X,Y,batch_size=1,nb_epoch=200,verbose=1,callbacks=[history])
    print("Epoch{:d}".format(i + 1))

lena_tmp = model.predict(X).reshape(lena.shape)
lena_tmp = np.uint8((lena_tmp - lena_tmp.min()) * 255 / (lena_tmp.max() - lena_tmp.min()))
Image.fromarray(lena_tmp).save("lena_sobel_stage_final.png")


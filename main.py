import os
import random
import shutil

import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_PATH = "/ai"
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# 创建一个templates（模板）对象，以后可以重用。
templates = Jinja2Templates(directory="templates")


@app.get(BASE_PATH)
async def root():
    return {"message": "Hello World"}


@app.get(BASE_PATH + "/predict")
async def pred(request: Request):
    mnist_test = pd.read_csv('mnist_test.csv')
    data_no = random.randint(0, 9999)
    im = mnist_test.iloc[data_no:data_no + 1, 1:]
    print(im.shape)
    image = im.values.reshape(28, 28)
    pyplot.imshow(image)
    file_name = 'static/images/number%s.jpg' % data_no;
    pyplot.savefig(file_name)

    model = load_model()
    input = mnist_test.iloc[data_no:data_no + 1, 1:]
    rst = model.predict(
        input, batch_size=None, verbose=0, steps=None, callbacks=None,
        max_queue_size=10,
        workers=1, use_multiprocessing=False
    )
    # pyplot.show()
    num = int(np.argmax(rst))
    url = str(request.url)
    url = url[0:url.find("ai/predict")] + file_name
    return {"img": url, "result": num}


@app.get(BASE_PATH + "/predict_test")
async def pred_test(request: Request):
    mnist_test = pd.read_csv('mnist_test.csv')
    data_no = random.randint(0, 9999)
    im = mnist_test.iloc[data_no:data_no + 1, 1:]
    print(im.shape)
    image = im.values.reshape(28, 28)
    pyplot.imshow(image)
    file_name = 'static/images/number%s.jpg' % data_no;
    pyplot.savefig(file_name)

    model = load_model()
    input = mnist_test.iloc[data_no:data_no + 1, 1:]
    rst = model.predict(
        input, batch_size=None, verbose=0, steps=None, callbacks=None,
        max_queue_size=10,
        workers=1, use_multiprocessing=False
    )
    # pyplot.show()
    num = int(np.argmax(rst))
    # return {"img": file_name, "result": num}
    return templates.TemplateResponse("predict.html", {"request": request, "img": "/" + file_name, "result": num})


@app.get(BASE_PATH + "/train")
async def do_train():
    train()
    return {"success": "OK"}


def showNumber(num):
    print('show number')
    mnist_test = pd.read_csv('mnist_test.csv')
    # print(mnist_test.info())
    print(mnist_test.shape)
    im = mnist_test.iloc[num:num + 1, 1:]
    print(im.shape)
    image = im.values.reshape(28, 28)
    print(image.shape)
    pyplot.imshow(image)
    pyplot.savefig('number%s.jpg' % num)
    pyplot.show()


def create_model():
    output_dim = 10
    hidden1_dim = 256
    hidden2_dim = 64

    model = tf.keras.Sequential([
        # 每张图片的大小是 28 * 28 = 784 像素, 这样 input_shape表示的是一个长度为784，宽度为1的已
        tf.keras.layers.Flatten(input_shape=(28 * 28, 1)),
        tf.keras.layers.Dense(hidden1_dim, activation='relu'),
        tf.keras.layers.Dense(hidden2_dim, activation='sigmoid'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def load_model():
    checkpoint_path = "training_model/cp.ckpt"
    model = create_model()
    model.load_weights(checkpoint_path)
    return model


def train():
    df_train = pd.read_csv('mnist_train.csv', header=None)
    df_test = pd.read_csv('mnist_test.csv', header=None)
    print(len(df_train), len(df_test))

    train_samples = len(df_train)
    test_samples = len(df_test)
    print(train_samples, test_samples)

    model = create_model()

    checkpoint_path = "training_model/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # 创建一个保存模型权重的回调
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    batch_size = 200
    train_batch_num = int(train_samples / batch_size)
    for epoch in range(10):
        print('begin train %s' % epoch)
        for batch_no in range(train_batch_num):
            start_no = batch_no * batch_size
            end_no = (batch_no + 1) * batch_size
            # 第二列到最后，是每张图的像素点
            batch_data = df_train.iloc[start_no:end_no, 1:]
            # 第一列，是每张图识别的数字
            batch_label = df_train.iloc[start_no:end_no, 0:1]

            model.fit(x=batch_data, y=batch_label, batch_size=batch_size, verbose=2,
                      callbacks=[cp_callback])

        print('end train %s' % epoch)

    test_data = df_test.iloc[0:test_samples, 1:]
    test_label = df_test.iloc[0:test_samples, 0:1]
    loss, acc = model.evaluate(x=test_data, y=test_label, verbose=1)
    print(loss, acc)


def evaluate():
    model = load_model()
    df_test = pd.read_csv('mnist_test.csv', header=None)
    test_samples = len(df_test)
    test_data = df_test.iloc[0:test_samples, 1:]
    test_label = df_test.iloc[0:test_samples, 0:1]
    loss, acc = model.evaluate(x=test_data, y=test_label, verbose=1)
    print(loss, acc)


def predict(data_no):
    mnist_test = pd.read_csv('mnist_test.csv')
    # print(mnist_test.info())
    print(mnist_test.shape)
    im = mnist_test.iloc[data_no:data_no + 1, 1:]
    print(im.shape)
    image = im.values.reshape(28, 28)
    pyplot.imshow(image)
    # pyplot.savefig('static/number%s.jpg' % data_no)
    # pyplot.show()

    model = load_model()
    input = mnist_test.iloc[data_no:data_no + 1, 1:]
    rst = model.predict(
        input, batch_size=None, verbose=0, steps=None, callbacks=None,
        max_queue_size=10,
        workers=1, use_multiprocessing=False
    )
    pyplot.show()
    print(rst)
    print(np.argmax(rst))
    num = int(np.argmax(rst))
    print('识别结果:%s' % num)


if __name__ == "__main__":
    print('remove test pics')
    # 删除临时文件夹
    shutil.rmtree("static/images", True)
    # 创建文件
    os.mkdir("static/images")
    print('run main')
    # showNumber(61)
    # train()
    # evaluate()
    # no = random.randint(0, 9999)
    # predict(no)
    uvicorn.run(app, host="0.0.0.0", port=9090)

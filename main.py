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


# 给前端进行预测调用的接口
@app.get("/api/predict")
async def pred(request: Request):
    # 加载测试数据
    mnist_test = pd.read_csv('mnist_test.csv')
    # 随机从1万张测试照片中选择一个行号，用来进行预测
    data_no = random.randint(0, 9999)
    im = mnist_test.iloc[data_no:data_no + 1, 1:]
    print(im.shape)
    image = im.values.reshape(28, 28)
    pyplot.imshow(image)
    file_name = 'static/images/number%s.jpg' % data_no;
    pyplot.savefig(file_name)

    model = load_model()
    input = mnist_test.iloc[data_no:data_no + 1, 1:]
    # 真正预测的函数 batch_size批量预测值，默认32；steps用于多步预测时使用，对于dataset类型忽略该参数使用；
    # callbacks在进行预测的时候，执行的回调函数；max_queue_size只针对Sequence的输入有效，生成队列的最大值；
    # workers只针对Sequence的输入有效，对于基于多线程的处理过程中，所运行使用处理器的最大值。
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


# 开发测试看看能不能生成图片
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


# 查看测试数据的图片
def show_number(num):
    print('show number')
    mnist_test = pd.read_csv('mnist_test.csv')
    # print(mnist_test.info())
    print(mnist_test.shape)
    # 去第num张的图片，由于下标从0开始，所以需要加一
    im = mnist_test.iloc[num:num + 1, 1:]
    print(im.shape)
    # 把一维数组变成28 * 28的二维数组，为后面变成图片准备
    image = im.values.reshape(28, 28)
    print(image.shape)
    # 使用二维数组显示图片
    pyplot.imshow(image)
    # 保存图片
    pyplot.savefig('number%s.jpg' % num)
    pyplot.show()


# 创建模型
def create_model():
    output_dim = 10
    hidden1_dim = 256
    hidden2_dim = 64

    model = tf.keras.Sequential([
        # 每张图片的大小是 28 * 28 = 784 像素, 这样 input_shape表示的把二维数组784 * 1转成一位数组，最后的1可以省略，但是逗号,不行
        tf.keras.layers.Flatten(input_shape=(28 * 28, 1)),
        # Dense是密集连接或全连接神经层，这个dense有256个节点，使用的激活函数是relu
        tf.keras.layers.Dense(hidden1_dim, activation='relu'),
        # 这个Dense是有64个节点，使用的激活函数是sigmoid
        tf.keras.layers.Dense(hidden2_dim, activation='sigmoid'),
        # 这个Dense层会返回一个长度为 10 的 logits 数组。每个节点都包含一个得分，用来表示当前图像属于 10 个类中的哪一类。
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])

    # 设置optimizer优化器, loss损失函数, metrics指标
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# 加载模型
def load_model():
    checkpoint_path = "training_model/cp.ckpt"
    model = create_model()
    # 仅读取权重 如果使用 load_model() 读取网络、权重
    model.load_weights(checkpoint_path)
    return model


# 训练模型
def train():
    # 加载训练数据
    df_train = pd.read_csv('mnist_train.csv', header=None)
    # 加载测试数据
    df_test = pd.read_csv('mnist_test.csv', header=None)
    print(len(df_train), len(df_test))

    train_samples = len(df_train)
    test_samples = len(df_test)
    print(train_samples, test_samples)

    # 加载模型
    model = create_model()

    checkpoint_path = "training_model/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # 创建一个保存模型权重的回调
    # 当使用model.fit来保存一个模型或权重的时候，可以使用这个保存上次训练状态的检查点文件，继续训练
    # save_weights_only=True，只保存权重值，否则整个模型都会保存
    # verbose=1 输出详细日志
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    # 设置每批训练200条数据
    batch_size = 200
    # 得到总的批次
    train_batch_num = int(train_samples / batch_size)
    # 设置训练10个完整周期
    for epoch in range(10):
        print('begin train %s' % epoch)
        for batch_no in range(train_batch_num):
            # 取每批数据的开始行号(从0开始计数)
            start_no = batch_no * batch_size
            # 取每批数据的结束行号
            end_no = (batch_no + 1) * batch_size
            # 第二列到最后，是每张图的像素点
            batch_data = df_train.iloc[start_no:end_no, 1:]
            # 第一列，是每张图识别的数字
            batch_label = df_train.iloc[start_no:end_no, 0:1]
            # 训练方法: x是输入数据，y是目标数据，batch_size是跑批大小，verbose  0 = silent, 1 = progress bar, 2 = one line per epoch
            # callbacks就是上面定义的回调方法 cp_callback
            model.fit(x=batch_data, y=batch_label, batch_size=batch_size, verbose=2,
                      callbacks=[cp_callback])

        print('end train %s' % epoch)

    # 取测试数据的第一列
    test_data = df_test.iloc[0:test_samples, 1:]
    # 取测试数据的第二列后的数据，就是一张图片的像素值
    test_label = df_test.iloc[0:test_samples, 0:1]
    # 比较模型在测试数据集上的表现 x输入测试数据，y测试目标数据
    # loss损失值，acc指标情况
    loss, acc = model.evaluate(x=test_data, y=test_label, verbose=1)
    print(loss, acc)


# 评估模型训练，具体介绍参见 train
def evaluate():
    model = load_model()
    df_test = pd.read_csv('mnist_test.csv', header=None)
    test_samples = len(df_test)
    test_data = df_test.iloc[0:test_samples, 1:]
    test_label = df_test.iloc[0:test_samples, 0:1]
    loss, acc = model.evaluate(x=test_data, y=test_label, verbose=1)
    print(loss, acc)


# 模型预测
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

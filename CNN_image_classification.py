import os

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn as sklearn
from PIL import Image

import re
import random

from concurrent.futures import ThreadPoolExecutor
import queue


from tkinter import *
import tkinter.filedialog
from PIL import Image, ImageTk
from tkinter.messagebox import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 公共常量
DEBUG = False
# 开发模式? 不修改代码并调试的话那设置这个为 False 即可

# 日志记录器
file = open("./log.txt", "a")

targets = [
        "Drought",
        "Fire_Disaster",
        "Land_Slide",
        "Water_Disaster",
        "Non_Damage"
]


# 工具函数
def gen_layer_name(where, role, num):
    # 此函数用于生成层的名字

    _layer_name = "{}_{}_{}".format(where, num, role)

    if DEBUG:
        _layer_name = "{}_{}".format(_layer_name, random.randint(0, 100000000))

    return _layer_name


def target_to_vector(target):
    """
    一共五种可能的取值:
        0. Drought: 干旱
        1. Fire_Disaster: 火灾
        2. Land_Slide: 山体滑坡
        3. Water_Disaster: 洪涝
        5. Non_Damage: 不是自然灾害
    :return: np.array
    """

    targets_index = {
        "Drought": 0,
        "Fire_Disaster": 1,
        "Land_Slide": 2,
        "Water_Disaster": 3,
        "Non_Damage": 4
    }

    vec = np.array([0, 0, 0, 0, 0])
    vec[targets_index[target]] = 1

    return vec


def vector_to_target(vec):
    return targets[np.argmax(vec)]


# 定义加载数据的函数
def data_reader():
    batch_size = 50
    start_index = 0

    # 存储多线程读取的数据的队列
    data_queue = queue.Queue(2)

    # 多线程读取数据
    pool = ThreadPoolExecutor(max_workers=1)
    future = None

    all_img_info = pd.read_csv("./Data/available/info.csv")
    all_img_info = all_img_info.sample(frac=1).reset_index(drop=True)

    def _read_data(q):
        # 读取数据 这个函数将在读取数据的线程中运行
        # q 即为队列对象

        nonlocal batch_size
        nonlocal start_index
        nonlocal all_img_info

        img_paths = all_img_info["path"]
        targets = all_img_info["type"]

        data_len = len(img_paths)

        # 准备存储数据的矩阵
        data_x = np.empty((0, 256, 256, 3))
        data_y = np.empty((0, 5))

        img_paths = img_paths[start_index: start_index + batch_size].reset_index(drop=True)
        targets = targets[start_index: start_index + batch_size].reset_index(drop=True)

        for img_index in range(len(img_paths)):
            img = Image.open(img_paths[img_index])
            img = np.asarray(img)

            im = img[np.newaxis, :, :, :]
            data_x = np.concatenate([data_x, im])

            target = targets[img_index]
            vector_target = target_to_vector(target)
            data_y = np.concatenate([data_y, vector_target[np.newaxis, :]])

        # 数据处理好后放到队列中
        q.put((data_x, data_y))

        # 如果数据都取完了就从头开始继续取
        if start_index + batch_size >= data_len:
            print("--- all data trained  now restart ---")
            file.write("--- all data trained  now restart ---\n")
            start_index = 0

            # 乱序训练数据
            all_img_info = all_img_info.sample(frac=1).reset_index(drop=True)
        else:
            start_index += batch_size

        return

    def _get_a_data():
        # 读取数据并控制读取线程的运行

        nonlocal future
        nonlocal data_queue
        nonlocal _read_data

        if data_queue.empty():
            # 当队列为空的时候有两种可能 要么线程已经开启 要么线程就没开

            if future is not None and future.running():
                # 当线程已经开启的时候 就直接阻塞等待即可
                while not future.done():
                    pass
            else:
                # 如果线程没开的话就开一个线程
                future = pool.submit(_read_data, data_queue)

            # 等待现有线程运行结束
            # 如果 "线程已经开启" 下的阻塞已经完成的话 那这里的阻塞就不会运行
            while not future.done():
                pass

            # 获取数据
            data = data_queue.get()

            # 然后继续新开一个线程即可
            future = pool.submit(_read_data, data_queue)

            return data

        # 队列不为空 直接取出数据返回即可
        data = data_queue.get()

        # 新开一个线程继续处理数据
        future = pool.submit(_read_data, data_queue)

        return data

    def _shutdown_pool():
        # 关闭线程池
        nonlocal pool

        pool.shutdown()
        return

    return _get_a_data, _shutdown_pool


# 生成神经网络层的函数
def gen_conv_layer(
        data,
        in_channels,
        out_channels,
        layer_No,
        use_maxpool=True,
        filter_width=3,
        conv_strides=[1, 1, 1, 1],
        pool_size=[1, 2, 2, 1],
        pool_strides=[1, 2, 2, 1],
):
    # 生成卷积层
    # data: 被卷积的数字
    # in_channels  输入的数据的通道数
    # out_channels  输出的数据的通道数(即输出几个数据)
    # layer_No  层编号
    # use_maxpool  是否使用 maxpool
    # filter_width  卷积核宽度 默认卷积核高度和宽度相同
    # conv_strides  卷积步幅
    # pool_size  池化核大小
    # pool_strides  池化步幅

    filter_ = tf.get_variable(
        gen_layer_name("convlayer", "filter", layer_No),
        shape=[filter_width, filter_width, in_channels, out_channels],
        dtype=np.float64,
        initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float64)
    )

    b = tf.get_variable(
        gen_layer_name("convlayer", "bias", layer_No),
        shape=[out_channels],
        dtype=np.float64,
        initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float64)
    )

    conv = tf.nn.conv2d(
        input=data,
        filter=filter_,
        strides=conv_strides,
        padding="SAME"
    )

    conv = tf.nn.bias_add(conv, b)

    conv = tf.nn.relu(conv)

    if use_maxpool:
        conv = tf.nn.max_pool(
            conv,
            ksize=pool_size,
            strides=pool_strides,
            padding="SAME"
        )

    return conv


def gen_fc_layer(
        data,
        in_size,
        out_size,
        layer_No,
        use_dropout=True,
        use_bais=True,
        use_relu=True,
        keep_prob=0.6
):
    # 生成全连接层的函数
    # data  输入数据数
    # in_size  输出神经元个数
    # out_size  输出神经元个数
    # layer_No  神经元层编号
    # use_dropout  是否随机丢弃某些神经元以防止过拟合
    # use_bais  是否添加偏置
    # keep_prob  在随机丢弃神经元操作中 每个神经元被保留下来的概率

    w = tf.get_variable(
        gen_layer_name("fc", "w", layer_No),
        shape=[in_size, out_size],
        dtype=np.float64,
        initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float64)
    )

    b = tf.get_variable(
        gen_layer_name("fc", "b", layer_No),
        shape=[out_size],
        dtype=np.float64,
        initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float64)
    )

    fc = tf.matmul(
        tf.reshape(
            data,
            shape=[-1, in_size]
        ),
        w
    )

    if use_bais:
        fc = tf.nn.bias_add(fc, b)

    if use_dropout:
        fc = tf.nn.dropout(
            fc,
            keep_prob=keep_prob
        )

    if use_relu:
        fc = tf.nn.relu(fc)

    return fc



'''
    模型结构:
    输入层 - 卷积层 - 输出层
    共 12 层 结构模仿 VGG 网络
    
    输入层
        输入 n 张 256x256x3 的图像
        
    卷积层
        输入 3 张图像(一张图像拆分为 RGB 通道后变为三张)
        输出 16 个卷积结果
        
    卷积层
        输入 16 个卷积结果 输出 16 个卷积结果
        
    池化层
        最大值池化 池化步幅 2x2  窗口大小 2x2
        
    卷积层
        输入 16 个卷积结果 输出 32 个卷积结果
        
    卷积层x3
        输入 32 个卷积结果 输出 32 个卷积结果
        
    池化层
        最大值池化 池化步幅 2x2  窗口大小 2x2
        
    全连接层
        输入 64*64*32 个数据 输出 512 个数据
        使用 relu
        dropout
        
    全连接层
        输入 512 个数据 输出 128 个数据
        使用 relu
        dropout
        
    输出层
        输入 128 个数据 输出 5
        不使用任何激活函数
        不使用 dropout
'''


# 声明模型
with tf.device("/gpu:0"):
    X = tf.placeholder(
        shape=[None, 256, 256, 3],
        dtype=tf.float64
    )

    Y = tf.placeholder(
        shape=[None, 5],
        dtype=tf.float64
    )

    conv1 = gen_conv_layer(X, 3, 16, layer_No=1, use_maxpool=False)
    conv2 = gen_conv_layer(conv1, 16, 16, layer_No=2, use_maxpool=False)

    maxpool_1 = tf.nn.max_pool(
        conv2,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME"
    )

    conv3 = gen_conv_layer(maxpool_1, 16, 32, layer_No=3, use_maxpool=False)
    conv4 = gen_conv_layer(conv3, 32, 32, layer_No=4, use_maxpool=False)
    conv5 = gen_conv_layer(conv4, 32, 32, layer_No=5, use_maxpool=False)
    conv6 = gen_conv_layer(conv5, 32, 32, layer_No=6, use_maxpool=False)

    maxpool_2 = tf.nn.max_pool(
        conv6,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME"
    )

    fc1 = gen_fc_layer(
        tf.reshape(maxpool_2, shape=[-1, 64 * 64 * 32]),
        64 * 64 * 32,
        512,
        layer_No=7
    )

    fc2 = gen_fc_layer(fc1, 512, 128, layer_No=8)

    out = gen_fc_layer(fc2, 128, 5, layer_No=9, use_dropout=False, use_bais=False, use_relu=False)


# 损失和优化
cost = tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=out,
    labels=Y
)

# 交叉熵
cost = tf.reduce_mean(cost)
# 交叉熵的平均值即为损失

op = tf.train.AdamOptimizer().minimize(cost)
# 使用 Adam 优化方法让损失最小


# 保存器 以及其他的一些需要的东西
gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=0.8
)
# 设置使用 GPU  并设置最多可以使用 GPU 的 80% 的内存

config = tf.ConfigProto(
    allow_soft_placement=True,
    gpu_options=gpu_options
)
# 生成配置对象: 允许使用 GPU  允许 tensorflow 自行根据情况使用设备
# 关闭输出日志 日志输出久了容易拖慢运行速度

saver = tf.train.Saver()
# 保存器 用于保存模型


# 获取数据
next_batch_data, shutdown_pool = data_reader()


# 训练
costs = []
accs = []
marco_precisions = []
marco_recalls = []
marco_f_measures = []
micro_f_measures = []
kappas = []
# 暂存数据的列表


# 训练的函数
def train():
    with tf.Session(config=config) as sess:
        if os.listdir("./Model"):
            # 如果存在训练过的模型 就加载训练过的模型然后继续训练

            models = os.listdir("./Model")

            models = [i.split(".")[0] for i in models]

            models = set(models)

            models = [
                (
                    int(re.findall(r"[a-zA-Z-_]+(\d+)", i)[0]),
                    i
                )
                for i in models
                if i != "checkpoint"
            ]

            models = sorted(models, key=lambda x: x[0])

            saver.restore(sess, save_path="./Model/" + models[-1][-1])
        else:
            # 否则就初始化全局变量即可
            sess.run(tf.global_variables_initializer())
            # 初始化全局变量

        train_count = 0
        # 训练计数器

        #     while acc < 0.9:
        #     while train_count < 20001:
        while True:
            Xtrain, Ytrain = next_batch_data()
            # 获取训练数据

            op_, cost_, pred = sess.run(
                fetches=[op, cost, out],
                feed_dict={
                    X: Xtrain.reshape(-1, 256, 256, 3),
                    Y: Ytrain
                }
            )
            # 执行优化操作 这一步就是训练
            # cost_ 是个数字  pred 是个 ndarr

            acc = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.argmax(pred, axis=-1),
                        tf.argmax(Ytrain, axis=-1)
                    ),
                    dtype=np.float64
                )
            )
            # 计算准确率
            # acc 是个 tf op

            if train_count % 50 == 0:
                # 每 50 次看一下准确率信息
                accval = sess.run(acc)

                print("--- count: {}  loss: {}  acc: {} ---".format(train_count, cost_, accval))

                file.write("count: {}  loss: {}  acc: {}\n".format(train_count, cost_, accval))
                file.flush()

            if train_count % 10 == 0:
                # 每 10 次训练暂存准确率信息

                accval = sess.run(acc)

                costs.append(cost_)
                accs.append(accval)

                Ytrain = np.argmax(Ytrain, axis=-1)
                pred = np.argmax(pred, axis=-1)

                marco_precisions.append(
                    sklearn.metrics.precision_score(Ytrain, pred, average="macro")
                )

                marco_recalls.append(
                    sklearn.metrics.recall_score(Ytrain, pred, average="macro")
                )

                marco_f_measures.append(
                    sklearn.metrics.f1_score(Ytrain, pred, average="macro")
                )

                micro_f_measures.append(
                    sklearn.metrics.f1_score(Ytrain, pred, average="micro")
                )

                kappas.append(
                    sklearn.metrics.cohen_kappa_score(Ytrain, pred)
                )

            if len(accs) == 100:
                # 保存准确率数据到硬盘 而后清空暂存数据
                # 这样一来就是每 10 * 100 = 1000 次训练就向硬盘保存一次信息

                np.save(
                    "./Log/Accs/Accs_{}".format(train_count),
                    np.array(accs)
                )

                np.save(
                    "./Log/Costs/Costs_{}".format(train_count),
                    np.array(costs)
                )

                np.save(
                    "./Log/Marco_precisions/Marco_precisions_{}".format(train_count),
                    np.array(marco_precisions)
                )

                np.save(
                    "./Log/Marco_recalls/Marco_recalls_{}".format(train_count),
                    np.array(marco_recalls)
                )

                np.save(
                    "./Log/Marco_f_measures/Marco_f_measures_{}".format(train_count),
                    np.array(marco_f_measures)
                )

                np.save(
                    "./Log/Micro_f_measures/Micro_f_measures_{}".format(train_count),
                    np.array(micro_f_measures)
                )

                np.save(
                    "./Log/Kappas/Kappas_{}".format(train_count),
                    np.array(kappas)
                )

                accs.clear()
                costs.clear()
                marco_precisions.clear()
                marco_recalls.clear()
                marco_f_measures.clear()
                micro_f_measures.clear()
                kappas.clear()
                # 保存完后就清空列表即可

            if train_count % 1000 == 0 and train_count != 0:
                # 每训练 1000 次保存一下模型
                saver.save(
                    sess,
                    save_path="./Model/image_classifcation_model",
                    global_step=train_count
                )

                file.write("saving the model: {}\n".format(train_count))

            train_count += 1


def test(x=None, y=None):
    print("testing...")

    with tf.Session(config=config) as sess:
        saver.restore(sess, save_path="./Model_done/image_classifcation_model-3000")

        if (x is not None) and (y is not None):
            Xtest = x
            Ytest = y
        else:
            Xtest, Ytest = next_batch_data()

        pred = sess.run(
            out,
            feed_dict={
                X: Xtest.reshape(-1, 256, 256, 3)
            }
        )

        for res_index in range(len(pred)):
            p = vector_to_target(pred[res_index])
            r = vector_to_target(Ytest[res_index])

            print("pred: {}  real: {}".format(p, r))

        pred = sess.run(tf.argmax(pred, axis=-1))
        real = sess.run(tf.argmax(Ytest, axis=-1))

        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(pred, real),
                dtype=np.float64
            )
        )

        print("average accuracy: {}".format(sess.run(acc)))

    return pred, real, acc


def select_img_to_predict(img_path):
    data_x = np.empty((0, 256, 256, 3))
    data_y = np.empty((0, 5))

    value = img_path.split(".")[-2]
    value = value.split("/")[-1]
    value = target_to_vector(value)
    value = value[np.newaxis, :]

    img = Image.open(img_path)
    img = np.asarray(img)
    img = img[np.newaxis, :, :, :]

    data_x = np.concatenate([data_x, img])
    data_y = np.concatenate([data_y, value])

    test(data_x, data_y)


def show_window():
    image_path = None

    window = Tk()
    status = Text(window)
    image_widget = Label(window)

    def start_action():
        nonlocal image_path
        nonlocal image_widget
        nonlocal status

        image_path = tkinter.filedialog.askopenfilename()
        # image_path = image_path.replace("/", "\\")
        status.insert("end", f"选择文件: {image_path}\n")

        img_open = Image.open(image_path)
        img_open = img_open.resize((200, 200))
        img_png = ImageTk.PhotoImage(img_open)
        image_widget.config(image=img_png)
        image_widget.image = img_png

        # 准备数据
        status.insert("end", f"读取数据\n")

        data_x = np.empty((0, 256, 256, 3))
        data_y = np.empty((0, 5))

        value = image_path.split(".")[-2]
        value = value.split("/")[-1]
        value = target_to_vector(value)
        value = value[np.newaxis, :]

        img = Image.open(image_path)
        img = np.asarray(img)
        img = img[np.newaxis, :, :, :]

        data_x = np.concatenate([data_x, img])
        data_y = np.concatenate([data_y, value])

        # 开始预测
        status.insert("end", f"开始预测\n")
        pred, real, acc = test(data_x, data_y)

        status.insert(
            "end",
            f"预测结果: 预测: {targets[pred[0]]}  实际: {targets[real[0]]} \n"
        )

        showinfo("预测结果", f"预测: {targets[pred[0]]}  实际: {targets[real[0]]}")

    window.title("Predict")

    status.pack()
    status.insert("end", "等待选择文件...\n")

    Button(window, text="路径选择", command=start_action).pack()

    image_widget.pack()

    def update_window():
        window.update()
        window.after(500, update_window)

    window.after(500, update_window)

    window.mainloop()


if __name__ == '__main__':
    RUN_TYPE = 2
    # 运行模式
    # 0 训练模式  1 测试模式  2 预测模式(指定图片路径进行预测)  3 显示一个窗口

    try:
        if RUN_TYPE == 0:
            train()
        elif RUN_TYPE == 1:
            test()
        elif RUN_TYPE == 2:
            select_img_to_predict("./Data/test/Drought.jpg")

        elif RUN_TYPE == 3:
            show_window()
    except Exception as e:
        file.close()
        shutdown_pool()

        print("exit. {}".format(e))


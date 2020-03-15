#!/usr/bin/env python
# coding: utf-8

# # 作业说明：
# 
# 1.请使用PaddlePaddle建立模型，对海洋鱼类数据进行分类，保证代码跑通。
# 
# 2.请考虑可以从哪些方面进行调整，评分标准为验证集上的准确率。

# # 图像分类是根据图像的语义信息将不同类别图像区分开来，是计算机视觉中重要的基本问题。
# 

# **1、准备数据**
# 
# **2、配置网络**
# 
# 	 （1）定义网络
#      
#      （2）定义损失函数
#      
#      （3）定义优化算法
#      
# **3、训练网络**
# 
# **4、模型评估**
# 
# **5、模型预测**

# In[1]:


#导入必要的包
import zipfile
import os
import random
import paddle
import sys
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import paddle.fluid as fluid
from multiprocessing import cpu_count
import matplotlib.pyplot as plt


# **数据集介绍**
# 
# 台湾电力公司、台湾海洋研究所和垦丁国家公园在2010年10月1日至2013年9月30日期间，在台湾南湾海峡、兰屿岛和胡比湖的水下观景台收集的鱼类图像数据集。
# 
# 该数据集包括23类鱼种，共27370张鱼的图像。
# 
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/b7213f7d65c54b6fae98d57de58e3d2ad29b68abc68140b4b283c615144cd40b)
# 
# 本实践选取5种鱼类数据作为数据集进行训练，被划分为两个子集，训练集和测试集比例为9:1
# 

# # **step1、准备数据：**
# 
# **1、生成数据列表**
# 
# **2、定义数据提供器 train_r、test_r**
# 
# **3、定义train_reader、test_reader**

# In[2]:


#解压原始数据集，将fish_image.zip解压至data目录下
src_path="/home/aistudio/data/data19062/fish_image.zip"
target_path="/home/aistudio/data/fish_image"
if(not os.path.isdir(target_path)):
    z = zipfile.ZipFile(src_path, 'r')
    z.extractall(path=target_path)
    z.close()


# In[3]:


#存放所有类别的信息
class_detail = []
#获取所有类别保存的文件夹名称
class_dirs = os.listdir(target_path+"/fish_image")
data_list_path="/home/aistudio/data/"

#每次执行代码，首先清空train.txt和test.txt
with open(data_list_path + "train.txt", 'w') as f: 
    pass
with open(data_list_path + "test.txt", 'w') as f: 
    pass


# In[4]:


#总的图像数量
all_class_images = 0
#存放类别标签
class_label=0
# 设置要生成文件的路径
data_root_path="/home/aistudio/data/fish_image/fish_image"
#存储要写进test.txt和train.txt中的内容
trainer_list=[]
test_list=[]
# 读取每个类别，['fish_01', 'fish_02', 'fish_03']
for class_dir in class_dirs:   
    #每个类别的信息
    class_detail_list = {}
    test_sum = 0
    trainer_sum = 0
    #统计每个类别有多少张图片
    class_sum = 0
    #获取类别路径 
    path = data_root_path + "/" + class_dir
    # 获取所有图片
    img_paths = os.listdir(path)
    for img_path in img_paths:                                  # 遍历文件夹下的每个图片
        name_path = path + '/' + img_path                       # 每张图片的路径
        if class_sum % 10 == 0:                                 # 每10张图片取一个做验证数据
            test_sum += 1                                       # test_sum为测试数据的数目
            test_list.append(name_path + "\t%d" % class_label + "\n")
        else:
            trainer_sum += 1 
            trainer_list.append(name_path + "\t%d" % class_label + "\n")#trainer_sum测试数据的数目
        class_sum += 1                                          #每类图片的数目
        all_class_images += 1                                   #所有类图片的数目
    class_label += 1  


random.shuffle(test_list)
with open(data_list_path + "test.txt", 'a') as f:
    for test in test_list:
        f.write(test) 
random.shuffle(trainer_list)

with open(data_list_path + "train.txt", 'a') as f:
    for train_image in trainer_list:
        f.write(train_image) 
print ('生成数据列表完成！')


# ![](https://ai-studio-static-online.cdn.bcebos.com/9d3504947dca47b7bb0bfd50700d4ff4c804e7c344be496bbdfc0d911e66b58f)

# In[5]:


#train_mapper/test_mapper 函数的作用是用来对训练集的图像进行处理修剪和数组变换，返回img和标签 
def train_mapper(sample):
    img_path, label = sample
    # 进行图片的读取，由于数据集的像素维度各不相同，需要进一步处理对图像进行变换
    img = paddle.dataset.image.load_image(img_path)       
    #进行了简单的图像变换，这里对图像进行crop修剪操作
    img = paddle.dataset.image.simple_transform(im=img,         
                                                resize_size=47, #缩放图片
                                                crop_size=47,   #剪裁
                                                is_color=True,   #是否彩色图像
                                                is_train=True)   #是否训练集
    #将img数组进行进行归一化处理，得到0到1之间的数值
    img= img.flatten().astype('float32')/255.0
    return img, label

# 定义获取人脸数据训练集的方法 train_r
def train_r(train_list, buffered_size=1024):
    def reader():
        with open(train_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab) 
    return paddle.reader.xmap_readers(train_mapper, reader,cpu_count(), buffered_size)  

def test_mapper(sample):
    img, label = sample
    img = paddle.dataset.image.load_image(img)
    img = paddle.dataset.image.simple_transform(im=img, 
                                                resize_size=47,
                                                crop_size=47,
                                                is_color=True,
                                                is_train=False)
    img= img.flatten().astype('float32')/255.0
    return img, label

# 对自定义数据集创建测试集提供器 test_r
def test_r(test_list, buffered_size=1024):
    def reader():
        with open(test_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)
    return paddle.reader.xmap_readers(test_mapper, reader,cpu_count(), buffered_size)


# In[6]:


BATCH_SIZE = 64
BUF_SIZE=512
#构造训练数据提供器
train_r= train_r(train_list= data_list_path + "train.txt")
train_reader = paddle.batch(
    paddle.reader.shuffle(
        reader=train_r,buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)

#构造测试数据提供器
test_r = test_r(test_list=data_list_path + "test.txt")
test_reader = paddle.batch(test_r,batch_size=BATCH_SIZE)


# # step2、配置网络
# 
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/1be52cfb6fc24d61995f9313990827a61bd9eb0d92d04eb39edb990770411875)
# 
# 
# **卷积层：执行卷积操作提取底层到高层的特征，发掘出图片“局部特性”；**
# 
# **池化层：通过降采样的方式，在不影响图像质量的情况下，压缩图片，减少参数；**
# 
# **全连接层：池化完成后，将数据“拍平”，丢到Flatten层，然后把Flatten层的输出放到全连接层里，可采用softmax对其进行分类。**
# 
# 

# In[7]:


def convolutional_neural_network(img):
    conv1=fluid.layers.conv2d(input=img,
							   num_filters=20,
							   filter_size=5,
							   act='relu')
    pool1=fluid.layers.pool2d(input=conv1,
							  pool_size=2,
							  pool_type='max',
							  pool_stride=2)
    conv_pool_1=fluid.layers.batch_norm(pool1)

    conv2=fluid.layers.conv2d(input=conv_pool_1,
							  num_filters=50,
							  filter_size=5,
							  act='relu')
    pool2=fluid.layers.pool2d(input=conv2,
							  pool_size=2,
							  pool_type='max',
							  pool_stride=2)
    conv_pool_2=fluid.layers.batch_norm(pool2)

    conv3=fluid.layers.conv2d(input=conv_pool_2,
							  num_filters=50,
							  filter_size=5,
							  act='relu')
    pool3=fluid.layers.pool2d(input=conv3,
							  pool_size=2,
							  pool_type='max',
							  pool_stride=2,
							  global_pooling=False)
    fc4=fluid.layers.fc(input=pool3,size=1024,act='relu')
    prediction=fluid.layers.fc(input=fc4,size=5,act='softmax')
    return prediction


# In[8]:


#定义两个张量
image=fluid.layers.data(name='image',shape=[3,47,47],dtype='float32')
label=fluid.layers.data(name='label',shape=[1],dtype='int64')


# In[9]:


#获取分类器
predict=convolutional_neural_network(image)


# **定义损失函数、优化方法**
# 
# 交叉熵损失函数在分类任务上比较常用。
# 
# 定义了一个损失函数之后，还要对它求平均值，因为定义的是一个Batch的损失值。
# 
# 同时我们还可以定义一个准确率函数，这个可以在我们训练的时候输出分类的准确率。
# 

# In[10]:


#定义损失函数
cost=fluid.layers.cross_entropy(input=predict,label=label)
avg_cost=fluid.layers.mean(cost)
accuracy=fluid.layers.accuracy(input=predict,label=label)


# In[11]:


#克隆main_program得到test_program，使用参数for_test来区分该程序是用来训练还是用来测试，该api请在optimization之前使用.
test_program = fluid.default_main_program().clone(for_test=True)


# In[12]:


#优化方法
optimizer=fluid.optimizer.Adam(learning_rate=0.0001)
optimizer.minimize(avg_cost)    

use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


# In[13]:





# **定义数据映射器**
# 
# DataFeeder负责将数据提供器（train_reader,test_reader）返回的数据转成一种特殊的数据结构，使其可以输入到Executor中。
# 
# feed_list设置向模型输入的向变量表或者变量表名

# In[13]:


#数据映射器
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])


# # step3、训练网络&step4、评估网络
# 

# In[14]:


#展示模型训练曲线
all_train_iter=0
all_train_iters=[]
all_train_costs=[]
all_train_accs=[]

all_test_iter=0
all_test_iters=[]
all_test_costs=[]
all_test_accs=[]

def draw_train_process(title,iters,costs,accs,label_cost,lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost) 
    plt.plot(iters, accs,color='green',label=lable_acc) 
    plt.legend()
    plt.grid()
    plt.show()


# In[15]:


EPOCH_NUM = 5
for pass_id in range(EPOCH_NUM):
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):                         #遍历train_reader的迭代器，并为数据加上索引batch_id
        train_cost, train_acc = exe.run(
            program=fluid.default_main_program(),                            #运行主程序
            feed=feeder.feed(data),                                          #喂入一个batch的数据
            fetch_list=[avg_cost, accuracy])                                 #fetch均方误差和准确率
    
        all_train_iter=all_train_iter+BATCH_SIZE
        all_train_iters.append(all_train_iter)
        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])
        
        if batch_id % 100 == 0:                                               #每100次batch打印一次训练、进行一次测试
            print("\nPass %d, Step %d, Cost %f, Acc %f" % 
            (pass_id, batch_id, train_cost[0], train_acc[0]))
            
    test_accs = []                                                            #测试的损失值
    test_costs = []                                                           #测试的准确率
    # 每训练一轮 进行一次测试
    for batch_id, data in enumerate(test_reader()):                           # 遍历test_reader
         test_cost, test_acc = exe.run(program=test_program,  # #运行测试主程序
                                       feed=feeder.feed(data),                #喂入一个batch的数据
                                       fetch_list=[avg_cost, accuracy])       #fetch均方误差、准确率
         test_accs.append(test_acc[0])                                        #记录每个batch的误差
         test_costs.append(test_cost[0])                                      #记录每个batch的准确率
         
         all_test_iter=all_test_iter+BATCH_SIZE
         all_test_iters.append(all_test_iter)
         all_test_costs.append(test_cost[0])                                       
         all_test_accs.append(test_acc[0])      

    test_cost = (sum(test_costs) / len(test_costs))                           # 每轮的平均误差
    test_acc = (sum(test_accs) / len(test_accs))                              # 每轮的平均准确率
    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))
    
    model_save_dir = "/home/aistudio/work/model"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # 保存训练的模型，executor 把所有相关参数保存到 dirname 中
    fluid.io.save_inference_model(model_save_dir,  #保存推理model的路径
                                  ['image'],       #推理（inference）需要 feed 的数据
                                  [predict],       #保存推理（inference）结果的 Variables
                                  exe)             #executor 保存 inference model
print('训练模型保存完成！')
draw_train_process("training",all_train_iters,all_train_costs,all_train_accs,"trainning cost","trainning acc")
draw_train_process("testing",all_test_iters,all_test_costs,all_test_accs,"testing cost","testing acc")


# # step5模型预测

# In[16]:


# 使用CPU进行训练
place = fluid.CPUPlace()
# 定义一个executor
infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()#要想运行一个网络，需要指明它运行所在的域，确切的说： exe.Run(&scope) 

#图片预处理
def load_image(file):
    im = Image.open(file)
    im = im.resize((47, 47), Image.ANTIALIAS)                 #resize image with high-quality 图像大小为28*28
    im = np.array(im).reshape(1, 3, 47, 47).astype(np.float32)#返回新形状的数组,把它变成一个 numpy 数组以匹配数据馈送格式。
    im = im / 255.0                            #归一化到【-1~1】之间
    return im

infer_img='/home/aistudio/data/data13981/00009598_05281.png'
#获取训练好的模型
#从指定目录中加载 推理model(inference model)
[inference_program,# 预测用的program
feed_target_names,# 是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。
fetch_targets] = fluid.io.load_inference_model(model_save_dir, infer_exe)#fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。

img = Image.open(infer_img)
plt.imshow(img)   #根据数组绘制图像
plt.show()        #显示图像

image=load_image(infer_img)

# 开始预测
results = infer_exe.run(
    inference_program,                      #运行预测程序
    feed={feed_target_names[0]: image},#喂入要预测的数据
    fetch_list=fetch_targets)               #得到推测结果
print('results',results)
label_list = [
        "Dascyllus reticulatus", "Plectrogly-phidodondickii", "Chromis chrysura", "Amephiprion clarkia", "Chaetodon lunulatus"
        ]
print("infer results: %s" % label_list[np.argmax(results[0])])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





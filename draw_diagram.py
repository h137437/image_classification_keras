from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os
import numpy as np

def read_tensorboard_data(tensorboard_path, val_name):
    """读取tensorboard数据，
    tensorboard_path是tensorboard数据地址，
    val_name是数据名称"""
    
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print(ea.scalars.Keys())
    val = ea.scalars.Items(val_name)
    return val

def draw_plt(vals, val_name, smooth_weight = 0.6):
    """将数据绘制成曲线图，
    vals是所有数据，
    val_name是变量名称"""
    
    plt.figure()
    for model_name, val in vals:
        steps = [i.step for i in val]
        values = [j.value for j in val]
        x = np.array(steps)
        y = np.array(values)
        y_smooth = smooth(y, smooth_weight)
        plt.plot(x, y_smooth, label=model_name)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(val_name)
    plt.savefig('.//'+val_name+'.png')
    plt.show()

def get_vals(rootdir, models, val_name):
    """得到所有的模型对应的训练参数，
    rootdir是存储tensorboard数据的目录，
    models是所有需要展示的模型名称对应的文件夹名，
    val_name是参数名"""

    vals = []
    for model in models:
        path = os.path.join(rootdir,model)
        filenames = os.listdir(path)
        for filename in filenames:
            if not filename.find('events'):
                print(filename)
                tensorboard_path = os.path.join(path,filename)
                val = read_tensorboard_data(tensorboard_path,val_name)
                vals.append((model,val))
    return vals

def smooth(datas, weight):
    """将僵硬的曲线转化成比较平滑的曲线，
    datas是要进行转化的数值"""
    
    smoothed = []
    last = datas[0]
    for data in datas:
        smoothed_val = last * weight + (1 - weight) * data
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


if __name__ == "__main__":
    rootdir = './/checkpoints//'
    models = {'InceptionV3-112','DenseNet-112','SEResNetXt-112','SEResNetXt-56'}
    vals = get_vals(rootdir,models,'loss')
    draw_plt(vals,'loss')
import os
from collections import Counter
import matplotlib.pyplot as plt
import math
import numpy as np


name_dict = {0: 'ignored regions', 1: 'pedestrian', 2: 'people',
             3: 'bicycle', 4: 'car', 5: 'van', 6: 'truck',
             7: 'tricycle', 8: 'awning-tricycle', 9: 'bus',
             10: 'motor', 11: 'others'}

pixels_dict = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}


# 标准2：coco数据集标准,小目标定义为分辨率小于32x32像素的。
def compute(path_annotations):
    file_anno = os.listdir(path_annotations)
    over_labels = 0
    small_labels = 0
    large_labels = 0
    occluded_labels = 0
    server_occluded_labels = 0
    occluded_small = 0
    server_occluded = 0
    k_list = []
    k_small = []
    o_list = []
    for f in file_anno:
        file_path_anno = os.path.join(path_annotations, f)
        anno_file = open(file_path_anno)
        anno_lines = anno_file.readlines()
        # over_labels = over_labels + len(anno_lines)
        for s in anno_lines:
            w = int(s.split(',')[2])
            h = int(s.split(',')[3])
            k = int(s.split(',')[5])
            o = int(s.split(',')[7])
            if k == 0 or k == 11:
                continue
            over_labels += 1
            k_list.append(k)
            pixels_dict[k].append(math.sqrt(w * h))
            if w * h <= 32 * 32:
                small_labels += 1
                k_small.append(k)
                if o > 0 :
                    occluded_small += 1
                if o == 2 :
                    server_occluded += 1
            if w * h >= 96 * 96:
                large_labels += 1
            if o > 0 :
                occluded_labels += 1
                o_list.append(k)
                if o == 2:
                    server_occluded_labels += 1
    kind = Counter(k_list)
    small_kind = Counter(k_small)
    occluded_kind = Counter(o_list)
    proportion_small = {category: small_kind[category] / total for category, total in kind.items()}
    proportion_occluded = {category: occluded_kind[category] / total for category, total in kind.items()}
    occulded_number = [over_labels - occluded_labels, occluded_labels - server_occluded_labels, server_occluded_labels]
    size_number = [small_labels, over_labels - small_labels - large_labels, large_labels]
    print("小目标占比：", small_labels / over_labels)
    print("小目标总数：", small_labels)
    print("遮挡目标总数：", occluded_labels)
    print("严重遮挡目标总数：", server_occluded_labels)
    print("遮挡目标占比：", occluded_labels / over_labels)
    print("遮挡小目标占比：", occluded_small / small_labels)
    print("严重遮挡小目标占比：", server_occluded / occluded_small)
    print("各类别小目标占比：", proportion_small)
    print("各类别遮挡目标占比：", proportion_occluded)
    print("各类别总数：", kind)
    print("各类别小目标总数：", small_kind)
    print("各类别遮挡目标总数：", occluded_kind)
    print("目标总数：", over_labels)
    return occluded_kind, kind


if __name__ == '__main__':
    print('训练集：')
    occluded, total = compute('/home/alex/alex/paper/code_cm/data/VisDrone2019-DET-train/annotations')
    # print('验证集：')
    # compute('/home/alex/alex/paper/code_cm/data/VisDrone2019-DET-val/annotations')
    # print('测试集：')
    # compute('/home/alex/alex/paper/code_cm/data/VisDrone2019-DET-test-dev/annotations')

    ## 目标种类分布图
    # keys = list(kind.keys())
    # names = [name_dict[key] for key in keys if key in name_dict]
    # values = list(kind.values())
    # plt.bar(names, values)
    # plt.title('Distribution of categories in training set')
    # plt.show()



    ## 目标尺度分布图
    # keys = list(pixels_dict.keys())
    # names = [name_dict[key] for key in keys if key in name_dict]
    # # x_loc = [1, 2, 3]
    # fig, ax = plt.subplots()
    # boxplot_data = list(pixels_dict.values())
    #
    # ax.boxplot(boxplot_data)
    # ax.set_xticklabels(names)
    # plt.title("Scale distribution of training set")
    # plt.ylim(0, 100)
    # plt.yticks(np.arange(0, 101, 10))
    # plt.xlabel("Category")
    # plt.ylabel("Average Pixels")
    # plt.show()


    ## 目标大小/遮挡情况饼图
    # labels = ['Unobscured', 'Partially obscured', 'Severely obscured']
    # labels = ['Small', 'Middle', 'Large']
    # colors = ['#ff9999', '#66b3ff', '#99ff99']  # 自定义颜色
    # explode = (0.1, 0, 0)  # 突出显示第一个类别（Category A）
    # plt.pie(size_number, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True)
    # plt.axis('equal')
    # plt.title('Proportion of object sizes')
    # plt.show()


    # 数据
    keys = sorted(total.keys())
    categories = [name_dict[key] for key in keys if key in name_dict]
    # 按排序后的键获取dict1和dict2的对应值
    all = [total[key] for key in keys]
    occlud = [occluded[key] for key in keys]

    # 堆叠条形图的高度
    bottom_partial = np.array(occlud)

    # 绘制图形
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制每个堆叠部分
    ax.bar(categories, all, label='Total', color='lightgrey')
    ax.bar(categories, occlud, label='Occlusion', color='skyblue', bottom=0)

    # 添加标签和标题
    # ax.set_ylabel('Count')
    ax.set_title('Occlusion distribution per category')
    ax.legend()

    # 显示图形
    plt.show()


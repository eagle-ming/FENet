# Object Detection in Aerial Images


# 所需环境
官方YOLOv8环境即可

# 文件下载

训练所需的权值可在百度网盘中下载。  
链接: https://pan.baidu.com/s/1IBiEkmjEnf5ElD2fzqRKIg?pwd=632d  
提取码: 632d 

VisDrone数据集下载地址如下：  
链接: https://pan.baidu.com/s/1NZp9E5cOOERbAnYj7cqo9A?pwd=318q   
提取码: 318q 


# 训练步骤

将yolov8_s.pth放入model_data，运行train.py  
具体训练步骤可参考[YOLOV8](https://github.com/bubbliiiing/yolov8-pytorch)


# 评估步骤

在FENet.py里面修改model_path, model_path指向训练好的权值文件，在logs文件夹里  
运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

Reference

https://github.com/bubbliiiing/yolov8-pytorch

航拍电网连接端识别检测数据集 yolo数据集 1200张![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d3d5c1219377472281afa14410630d7d.png)

标注名称以及数量:
5：3302
6：1287
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3b7eb17c445e46f29c2cd6fce2fdf069.png)
航拍电网连接端识别检测![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9278622af0af448eae599fcb7849cc81.png)
航拍电网连接端识别检测![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2bd87bb6bb66485685f8924fb3ed3edf.png)
基于航拍电网连接端识别检测数据集进行目标检测任务。数据集包含1200张图片，并且已经标注为YOLO格式。具体类别及其数量如下：
- 类别5: 3302个实例
- 类别6: 1287个实例

## 备注：文章代码仅供参考！

### 环境准备

确保您已经安装了以下软件和库：
- Python 3.8 或更高版本
- PyTorch 1.9 或更高版本
- torchvision 0.10 或更高版本
- OpenCV
- numpy
- pandas
- matplotlib
- albumentations（用于数据增强）

您可以使用以下命令安装所需的Python库：

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib albumentations
```

### 数据集准备

假设您的数据集已经按照YOLO格式组织好，并且包含训练集、验证集和测试集。以下是数据集的预期结构：

```
datasets/
└── grid_connection_points/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

同时，有一个 `classes.txt` 文件包含类别名称，每行一个类别名称。

#### 类别文件 (`classes.txt`)

```
connection_point_5
connection_point_6
```

### 模型训练

我们将使用YOLOv5进行训练。首先，克隆YOLOv5仓库并设置环境。

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

#### 准备配置文件

创建一个 `hyp.scratch.yaml` 文件来定义超参数：

```yaml
# Hyperparameters for YOLOv5 training from scratch
lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.1  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.5  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.0  # image mixup (probability)
copy_paste: 0.0  # segment copy-paste (probability)
paste_in: 0.0  # segment paste-in (probability)
rect: 0  # rectangular training
resume: false  # resume training from last checkpoint
nosave: false  # only save final checkpoint
noval: false  # only validate final epoch
noautoanchor: true  # disable AutoAnchor
evolve: false  # evolve hyperparameters
bucket: ''  # gsutil bucket
cache_images: false  # cache images for faster training
image_weights: false  # use weighted image selection for training
single_cls: false  # train multi-class data as single-class
optimizer: SGD  # optimizer: SGD or AdamW
sync_bn: false  # use SyncBatchNorm, only available in DDP mode
workers: 8  # dataloader workers (max is number of CPU cores)
freeze: 0  # freeze first n layers
v5_metric: true  # assume maximum recall as 1.0 in AP calculation
multi_scale: true  # vary input size between 320 and 640 pixels
rect_training: false  # rectangular training
cos_lr: false  # cosine LR scheduler
close_mosaic: 1000  # close mosaic border
scales: [0.5, 1.5]  # image size scales
augment: true  # augment data
verbose: false  # verbose print
seed: 0  # random seed for reproducibility
local_rank: -1  # ddp device id (-1 for single gpu train)
entity: null  # W&B entity
upload_dataset: False  # upload dataset as W&B artifact table
bbox_interval: -1  # W&B bounding box logging interval
artifact_alias: latest  # version of dataset artifact to use
project: runs/train  # save results to project/name
exist_ok: false  # existing project/name ok, do not increment
quad: false  # quadrilateral anchors
linear_assignment: false  # use linear assignment for NMS
```

创建一个 `data.yaml` 文件来定义数据集路径和类别：

```yaml
train: ../datasets/grid_connection_points/images/train/
val: ../datasets/grid_connection_points/images/val/

nc: 2  # number of classes
names: ['connection_point_5', 'connection_point_6']  # list of class names
```

#### 训练模型

使用以下命令开始训练：

```bash
python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --cfg yolov5s.yaml --weights yolov5s.pt --hyp hyp.scratch.yaml
```

### 结果评估

训练完成后，可以使用以下命令评估模型性能：

```bash
python val.py --data data.yaml --weights runs/train/exp/weights/best.pt --task test
```

### 使用说明

1. **配置路径**：
   - 确保 `datasets/grid_connection_points/` 目录结构正确。
   - 确保 `data.yaml` 中的路径和类别名称正确。

2. **运行脚本**：
   - 在终端中依次运行训练脚本和评估脚本。

3. **注意事项**：
   - 根据需要调整超参数和训练设置。
   - 可以通过修改 `data.yaml` 中的 `cfg` 参数来选择不同的YOLOv5模型架构（如 `yolov5m.yaml`, `yolov5l.yaml`, `yolov5x.yaml`）。

### 示例

假设您的数据集文件夹结构如下：
```
datasets/
└── grid_connection_points/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

并且 `images/` 和 `labels/` 目录分别包含对应的图片和标注文件。运行上述脚本后，您可以查看训练日志和最终的模型权重文件。

### 总结



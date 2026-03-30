import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
from ultralytics.utils.downloads import download
from ultralytics.utils import ASSETS_URL, TQDM

# 手动定义 yaml 字典，补齐代码引用的变量
yaml = {
    "path": "./datasets/VOC",
    "names": {
        0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 
        5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 
        10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 
        15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
    }
}

def convert_label(path, lb_path, year, image_id):
    """将 VOC XML 标注转换为归一化的 [class_id, xmin, ymin, xmax, ymax] 格式"""

    def convert_to_minmax(size, box):
        """
        size: (width, height)
        box: [xmin, xmax, ymin, ymax] (VOC 原始像素坐标)
        返回: (xmin_norm, ymin_norm, xmax_norm, ymax_norm)
        """
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        
        # 直接归一化原始坐标
        xmin_n = box[0] * dw
        xmax_n = box[1] * dw
        ymin_n = box[2] * dh
        ymax_n = box[3] * dh
        
        # 限制在 [0, 1] 范围内，防止微小溢出
        return (max(0, min(1, xmin_n)), 
                max(0, min(1, ymin_n)), 
                max(0, min(1, xmax_n)), 
                max(0, min(1, ymax_n)))

    with open(path / f"VOC{year}/Annotations/{image_id}.xml") as in_file, \
         open(lb_path, "w", encoding="utf-8") as out_file:
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        names = list(yaml["names"].values())
        for obj in root.iter("object"):
            cls = obj.find("name").text
            # 过滤掉不在名单内或标记为 difficult 的物体
            if cls in names and int(obj.find("difficult").text) != 1:
                xmlbox = obj.find("bndbox")
                # 提取原始像素坐标
                pixel_box = [
                    float(xmlbox.find("xmin").text),
                    float(xmlbox.find("xmax").text),
                    float(xmlbox.find("ymin").text),
                    float(xmlbox.find("ymax").text)
                ]
                
                # 转换
                norm_box = convert_to_minmax((w, h), pixel_box)
                cls_id = names.index(cls)
                
                # 写入格式: class_id xmin ymin xmax ymax
                out_file.write(f"{cls_id} {' '.join(f'{a:.6f}' for a in norm_box)}\n")


  # Download
# 方案 A：相对路径（推荐，方便代码迁移）
dir = Path("./datasets/VOC")
urls = [
      f"{ASSETS_URL}/VOCtrainval_06-Nov-2007.zip",  # 446MB, 5012 images
      f"{ASSETS_URL}/VOCtest_06-Nov-2007.zip",  # 438MB, 4953 images
      f"{ASSETS_URL}/VOCtrainval_11-May-2012.zip",  # 1.95GB, 17126 images
  ]
download(urls, dir=dir / "images", threads=3, exist_ok=True)  # download and unzip over existing (required)

  # Convert
path = dir / "images/VOCdevkit"
for year, image_set in ("2012", "train"), ("2012", "val"), ("2007", "train"), ("2007", "val"), ("2007", "test"):
      imgs_path = dir / "images" / f"{image_set}{year}"
      lbs_path = dir / "labels" / f"{image_set}{year}"
      imgs_path.mkdir(exist_ok=True, parents=True)
      lbs_path.mkdir(exist_ok=True, parents=True)

      with open(path / f"VOC{year}/ImageSets/Main/{image_set}.txt") as f:
          image_ids = f.read().strip().split()
      for id in TQDM(image_ids, desc=f"{image_set}{year}"):
          f = path / f"VOC{year}/JPEGImages/{id}.jpg"  # old img path
          lb_path = (lbs_path / f.name).with_suffix(".txt")  # new label path
          f.rename(imgs_path / f.name)  # move image
          convert_label(path, lb_path, year, id)  # convert labels to YOLO format

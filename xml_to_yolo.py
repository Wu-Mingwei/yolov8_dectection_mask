import os
import xmltodict
from tqdm import tqdm
import cv2

# 输入路径
# xml_dir = r"C:\Users\wming\OneDrive\Desktop\kaggle\yolov8\datasets\labels\trains"
# image_dir = r"C:\Users\wming\OneDrive\Desktop\kaggle\yolov8\datasets\images\trains"
# output_dir = r"C:\Users\wming\OneDrive\Desktop\kaggle\yolov8\datasets\annotations\trains"

xml_dir = r"C:\Users\wming\OneDrive\Desktop\kaggle\yolov8\datasets\labels\vals"
image_dir = r"C:\Users\wming\OneDrive\Desktop\kaggle\yolov8\datasets\images\vals"
output_dir = r"C:\Users\wming\OneDrive\Desktop\kaggle\yolov8\datasets\annotations\vals"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 类别映射
class_names = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect" : 2}

for xml_file in tqdm(os.listdir(xml_dir)):
    if not xml_file.endswith(".xml"):
        continue 

    # 解析XML
    with open(os.path.join(xml_dir, xml_file), "r") as f:
        xml_data = xmltodict.parse(f.read())

    img_path = os.path.join(image_dir, xml_file.replace(".xml", ".png"))
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告：无法读取图片 {img_path}，跳过此文件")
        continue
    img_h, img_w = img.shape[:2]

    txt_lines = []
    objects = xml_data["annotation"]["object"]
    if not isinstance(objects, list):
        objects = [objects]

    for obj in objects:
        class_name = obj["name"]
        try:
            class_id = class_names[class_name]
        except KeyError:
            print(f"错误：未知类别 '{class_name}'，请检查 class_names 映射")
            continue

        # 解析边界框
        bbox = obj["bndbox"]
        x_min = float(bbox["xmin"])
        y_min = float(bbox["ymin"])
        x_max = float(bbox["xmax"])
        y_max = float(bbox["ymax"])

        # 计算YOLO格式坐标（归一化）
        x_center = (x_min + x_max) / 2 / img_w
        y_center = (y_min + y_max) / 2 / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h

        # 添加到结果列表（确保数值在[0,1]范围内）
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        txt_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 写入TXT文件（整个XML处理完成后）
    if txt_lines:
        txt_file = xml_file.replace(".xml", ".txt")
        with open(os.path.join(output_dir, txt_file), "w") as f:
            f.write("\n".join(txt_lines))
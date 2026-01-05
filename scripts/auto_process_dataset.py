import os
import shutil
import random
import yaml
import json
import cv2
from tqdm import tqdm

# 加载配置
def load_total_config(config_path="./config/total_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 1. 解析路径（仅读取指定的单个目录）
    root_path = os.path.abspath(config['paths']['root_path'])
    target_scene_dir = os.path.abspath(config['paths']['target_scene_dir'])  # 单个目标目录
    output_root = os.path.abspath(config['paths']['output_root'])
    
    # 2. 解析类别映射
    class_dict = config['dataset']['class_dict']
    cn2en = class_dict
    en2cid = {en_name: idx for idx, en_name in enumerate(class_dict.values())}
    
    return {
        "target_scene_dir": target_scene_dir,  # 仅处理这个目录
        "output_root": output_root,
        "train_path": os.path.join(output_root, 'train'),
        "val_path": os.path.join(output_root, 'val'),
        "cn2en": cn2en,
        "en2cid": en2cid,
        "class_num": len(class_dict),
        "train_ratio": config['dataset']['train_ratio'],
        "random_seed": config['dataset']['random_seed'],
        "img_formats": config['dataset']['img_formats'],
        "delete_temp": config['dataset']['delete_temp_files'],
        "config_path": config_path
    }

# 扫描指定的单个目录，收集<图像, JSON>配对
def scan_target_dir(config):
    file_pairs = []
    scene_path = config['target_scene_dir']
    
    # 检查指定目录是否存在
    if not os.path.exists(scene_path):
        print(f"❌ 错误：指定的目录不存在 → {scene_path}")
        return []
    
    print(f"✅ 开始处理指定目录：{scene_path}")
    
    # 仅遍历该目录下的图像文件
    img_files = [f for f in os.listdir(scene_path) 
                 if any(f.lower().endswith(fmt) for fmt in config['img_formats'])]
    
    if not img_files:
        print(f"❌ {scene_path} 下未找到任何图像文件（支持格式：{config['img_formats']}）")
        return []
    
    # 匹配图像和对应的JSON标注
    for img_file in img_files:
        img_path = os.path.join(scene_path, img_file)
        json_name = os.path.splitext(img_file)[0] + '.json'
        json_path = os.path.join(scene_path, json_name)
        
        if os.path.exists(json_path):
            file_pairs.append((img_path, json_path))
        else:
            print(f"⚠️ {img_path} 无对应JSON标注，跳过")
    
    if not file_pairs:
        print("❌ 未找到任何<图像+JSON>配对文件")
        return []
    
    print(f"✅ 共找到 {len(file_pairs)} 组有效数据")
    return file_pairs

# LabelMe JSON转YOLO TXT标签（逻辑不变）
def json2yolo(img_path, json_path, save_label_path, config):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 无法读取图像 {img_path}，跳过")
            return False
        img_h, img_w = img.shape[:2]
        
        txt_lines = []
        for shape in data['shapes']:
            original_cn_label = shape['label'].strip()
            
            if original_cn_label not in config['cn2en']:
                print(f"⚠️ {json_path} 中未知类别：{original_cn_label}，跳过该标注")
                continue
            en_label = config['cn2en'][original_cn_label]
            cid = config['en2cid'].get(en_label, -1)
            if cid == -1:
                print(f"⚠️ {json_path} 中 {original_cn_label} → {en_label} 无对应ID，跳过该标注")
                continue
            
            points = shape['points']
            if len(points) < 2:
                print(f"⚠️ {json_path} 中 {original_cn_label} 标注点数量不足，跳过该标注")
                continue
            
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            xmin = min(x_coords)
            ymin = min(y_coords)
            xmax = max(x_coords)
            ymax = max(y_coords)
            
            if xmin >= xmax or ymin >= ymax:
                print(f"⚠️ {json_path} 中 {original_cn_label} 标注框无效，跳过该标注")
                continue
            
            x_center = (xmin + xmax) / 2 / img_w
            y_center = (ymin + ymax) / 2 / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h
            
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                print(f"⚠️ {json_path} 中 {original_cn_label} 坐标超出范围，跳过该标注")
                continue
            
            txt_lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        txt_lines = list(set(txt_lines))
        
        with open(save_label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt_lines))
        
        if not txt_lines:
            print(f"⚠️ {json_path} 转换后无有效标注，生成空标签文件")
        
        return True
    except Exception as e:
        print(f"❌ 转换 {json_path} 失败：{str(e)}")
        return False

# 清空目录
def clear_dir(dir_path):
    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

# 主处理逻辑
def auto_process_dataset():
    config = load_total_config()
    
    # 仅扫描指定的单个目录
    file_pairs = scan_target_dir(config)
    if not file_pairs:
        return False
    
    # 初始化输出目录
    train_img_dir = os.path.join(config['train_path'], 'images')
    train_label_dir = os.path.join(config['train_path'], 'labels')
    val_img_dir = os.path.join(config['val_path'], 'images')
    val_label_dir = os.path.join(config['val_path'], 'labels')
    
    # 创建目录并清空原有内容
    for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        os.makedirs(dir_path, exist_ok=True)
        clear_dir(dir_path)
    
    # 划分训练/验证集
    random.seed(config['random_seed'])
    random.shuffle(file_pairs)
    train_num = int(len(file_pairs) * config['train_ratio'])
    train_pairs = file_pairs[:train_num]
    val_pairs = file_pairs[train_num:]
    
    print(f"\n📊 数据集划分：训练集 {len(train_pairs)} 张，验证集 {len(val_pairs)} 张")
    
    # 处理训练集
    print("\n=== 处理训练集 ===")
    train_success = 0
    for img_path, json_path in tqdm(train_pairs, desc="训练集转换"):
        img_name = os.path.basename(img_path)
        dst_img_path = os.path.join(train_img_dir, img_name)
        shutil.copy(img_path, dst_img_path)
        
        label_name = os.path.splitext(img_name)[0] + '.txt'
        dst_label_path = os.path.join(train_label_dir, label_name)
        if json2yolo(img_path, json_path, dst_label_path, config):
            train_success += 1
    
    # 处理验证集
    print("\n=== 处理验证集 ===")
    val_success = 0
    for img_path, json_path in tqdm(val_pairs, desc="验证集转换"):
        img_name = os.path.basename(img_path)
        dst_img_path = os.path.join(val_img_dir, img_name)
        shutil.copy(img_path, dst_img_path)
        
        label_name = os.path.splitext(img_name)[0] + '.txt'
        dst_label_path = os.path.join(val_label_dir, label_name)
        if json2yolo(img_path, json_path, dst_label_path, config):
            val_success += 1
    
    # 输出统计结果
    print(f"\n✅ 数据集处理完成：")
    print(f"  训练集：成功转换 {train_success}/{len(train_pairs)} 张")
    print(f"  验证集：成功转换 {val_success}/{len(val_pairs)} 张")
    print(f"  训练集图像：{train_img_dir}")
    print(f"  训练集标签：{train_label_dir}")
    print(f"  验证集图像：{val_img_dir}")
    print(f"  验证集标签：{val_label_dir}")
    
    # 更新配置文件中的train/val路径
    with open(config['config_path'], 'r+', encoding='utf-8') as f:
        total_config = yaml.safe_load(f)
        total_config['paths']['train'] = config['train_path']
        total_config['paths']['val'] = config['val_path']
        f.seek(0)
        yaml.dump(total_config, f, indent=2, allow_unicode=True)
        f.truncate()
    
    return True

if __name__ == "__main__":
    auto_process_dataset()
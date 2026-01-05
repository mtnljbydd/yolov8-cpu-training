import os
import shutil
import yaml
import re
import sys
from tqdm import tqdm
# 彻底解决Windows/CPU下进度条换行问题的核心配置
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TQDM_DISABLE'] = 'False'
os.environ['TQDM_POSITION'] = '0'
os.environ['TQDM_NCOLS'] = '100'
os.environ['TQDM_LINE_BREAKS'] = 'False'
os.environ['ULTRALYTICS_VERBOSE'] = 'True'
# 强制使用兼容Windows的进度条渲染器
os.environ['TQDM_ENV'] = 'windows'

from ultralytics import YOLO
from ultralytics.utils import LOGGER

# 重写tqdm类，强制单行更新
class SingleLineTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs['dynamic_ncols'] = True
        kwargs['position'] = 0
        kwargs['leave'] = False  # 进度条完成后不保留，仅Epoch结束后显示汇总
        kwargs['ncols'] = 100
        kwargs['bar_format'] = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        super().__init__(*args, **kwargs)
    
    def update(self, n=1):
        # 强制单行刷新，不换行
        super().update(n)
        self.refresh()

# 替换ultralytics内部的tqdm
import ultralytics.utils.torch_utils
ultralytics.utils.torch_utils.tqdm = SingleLineTqdm
import ultralytics.engine.trainer
ultralytics.engine.trainer.tqdm = SingleLineTqdm

# 调整日志级别，保留核心信息（仅保留合法配置）
LOGGER.setLevel('INFO')

# 加载配置（适配target_scene_dir）
def load_total_config(config_path="./config/total_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 1. 解析模型命名配置
    model_naming = config['model_naming']
    custom_model_name = model_naming['model_name']
    custom_model_version = model_naming['model_version']
    final_exp_name = f"{custom_model_name}_{custom_model_version}"
    
    # 2. 解析路径配置
    root_path = os.path.abspath(config['paths']['root_path'])
    target_scene_dir = os.path.abspath(config['paths']['target_scene_dir'])
    output_root = os.path.abspath(config['paths']['output_root'])
    train_path = os.path.abspath(config['paths'].get('train', os.path.join(output_root, 'train')))
    val_path = os.path.abspath(config['paths'].get('val', os.path.join(output_root, 'val')))
    
    # 替换导出路径变量
    export_save_path = config['paths']['export_save_path'].replace("{root_path}", root_path)
    export_save_path = export_save_path.replace("{model_name}", custom_model_name)
    export_save_path = export_save_path.replace("{model_version}", custom_model_version)
    export_save_path = os.path.abspath(export_save_path)
    
    # 3. 解析类别信息
    class_dict = config['dataset']['class_dict']
    en2cid = {en_name: idx for idx, en_name in enumerate(class_dict.values())}
    sorted_en_names = [en for en, cid in sorted(en2cid.items(), key=lambda x: x[1])]
    
    return {
        "train_path": train_path,
        "val_path": val_path,
        "target_scene_dir": target_scene_dir,
        "nc": len(class_dict),
        "names": sorted_en_names,
        "custom_model_name": custom_model_name,
        "custom_model_version": custom_model_version,
        "final_exp_name": final_exp_name,
        "export_save_path": export_save_path,
        "model": config['training']['model'],
        "epochs": config['training']['epochs'],
        "batch_size": config['training']['batch'],
        "imgsz": config['training']['imgsz'],
        "device": config['training']['device'],
        "patience": config['training']['patience'],
        "save_period": config['training']['save_period'],
        "lr0": config['training']['learning_rate'],
        "weight_decay": config['training']['weight_decay'],
        "momentum": config['training']['momentum'],
        "warmup_epochs": config['training']['warmup_epochs'],
        "project": config['training']['project'],
        "exist_ok": config['training']['exist_ok'],
        "conf": config['validation']['conf'],
        "iou": config['validation']['iou'],
        "save_json": config['validation']['save_json'],
        "plots": config['validation']['plots'],
        "delete_temp": config['dataset']['delete_temp_files'],
        "output_root": output_root,
        "config_path": config_path
    }

# 生成YOLO所需的临时data.yaml
def generate_yolo_data_yaml(config):
    temp_data = {
        "train": config['train_path'],
        "val": config['val_path'],
        "nc": config['nc'],
        "names": config['names']
    }
    
    temp_data_path = "./config/temp_yolo_data.yaml"
    os.makedirs(os.path.dirname(temp_data_path), exist_ok=True)
    with open(temp_data_path, 'w', encoding='utf-8') as f:
        yaml.dump(temp_data, f, indent=2, allow_unicode=True)
    
    return temp_data_path

# 重命名训练后的模型文件
def rename_trained_models(config):
    original_exp_dir = os.path.join(config['project'], "temp")
    final_exp_dir = os.path.join(config['project'], config['final_exp_name'])
    
    if os.path.exists(original_exp_dir) and not os.path.exists(final_exp_dir):
        os.rename(original_exp_dir, final_exp_dir)
        print(f"✅ 实验目录已重命名：{original_exp_dir} → {final_exp_dir}")
    
    weights_dir = os.path.join(final_exp_dir, "weights")
    if os.path.exists(weights_dir):
        for file_name in os.listdir(weights_dir):
            if file_name in ["best.pt", "last.pt"]:
                new_file_name = f"{config['custom_model_name']}_{config['custom_model_version']}_{file_name}"
                old_path = os.path.join(weights_dir, file_name)
                new_path = os.path.join(weights_dir, new_file_name)
                
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"✅ 模型文件已重命名：{old_path} → {new_path}")
    
    os.makedirs(config['export_save_path'], exist_ok=True)
    print(f"✅ 模型导出目录已创建：{config['export_save_path']}")
    
    return final_exp_dir

# 一键训练YOLOv8
def train_yolov8():
    config = load_total_config()
    
    # 生成临时data.yaml
    temp_data_path = generate_yolo_data_yaml(config)
    print(f"✅ 生成临时data.yaml：{temp_data_path}")
    
    # 加载预训练模型
    model = YOLO(config['model'])
    print(f"✅ 加载模型：{config['model']}")
    print(f"✅ 训练设备：{config['device']}")
    print(f"✅ 训练轮数：{config['epochs']}")
    print(f"✅ 批次大小：{config['batch_size']}")
    print(f"✅ 类别数量：{config['nc']}")
    print(f"✅ 训练数据来源：{config['target_scene_dir']}")
    
    # 开始训练（核心：解决进度条换行问题）
    print("\n=== 开始YOLOv8训练 ===")
    results = model.train(
        data=temp_data_path,
        epochs=config['epochs'],
        batch=config['batch_size'],
        imgsz=config['imgsz'],
        device=config['device'],
        patience=config['patience'],
        save=True,
        save_period=config['save_period'],
        lr0=config['lr0'],
        weight_decay=config['weight_decay'],
        momentum=config['momentum'],
        warmup_epochs=config['warmup_epochs'],
        val=True,
        cache='disk',  # 改用disk缓存，消除RAM缓存警告
        verbose=True,
        project=config['project'],
        name="temp",
        exist_ok=config['exist_ok'],
        plots=config['plots'],
        save_json=config['save_json'],
        workers=0,  # CPU训练强制workers=0，避免多线程进度条错乱
        single_cls=False
    )
    
    # 重命名模型和实验目录
    final_exp_dir = rename_trained_models(config)
    
    # 验证最佳模型
    print("\n=== 验证最佳模型 ===")
    best_model_name = f"{config['custom_model_name']}_{config['custom_model_version']}_best.pt"
    best_model_path = os.path.join(final_exp_dir, "weights", best_model_name)
    
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(final_exp_dir, "weights", "best.pt")
        if not os.path.exists(best_model_path):
            print(f"❌ 未找到最佳模型：{best_model_path}")
            return
    
    best_model = YOLO(best_model_path)
    val_results = best_model.val(
        data=temp_data_path,
        conf=config['conf'],
        iou=config['iou'],
        save_json=config['save_json'],
        plots=config['plots'],
        verbose=False
    )
    
    # 输出核心结果
    print(f"\n🎉 训练+验证完成！")
    print(f"📁 实验目录：{final_exp_dir}")
    print(f"📌 最佳模型路径：{best_model_path}")
    print(f"📊 核心指标：")
    print(f"   - 验证集mAP@0.5：{val_results.box.map50:.4f}（越接近1越好）")
    print(f"   - 验证集mAP@0.5:0.95：{val_results.box.map:.4f}")
    try:
        print(f"   - 最终box_loss：{results.results_dict['train/box_loss']:.4f}（越接近0越好）")
        print(f"   - 最终cls_loss：{results.results_dict['train/cls_loss']:.4f}（越接近0越好）")
    except KeyError:
        print(f"   - 最终训练损失：参考 runs/detect/{config['final_exp_name']}/results.csv")
    print(f"📤 模型导出目录：{config['export_save_path']}")
    
    # 可选删除临时文件
    if config['delete_temp']:
        if os.path.exists(config['output_root']):
            shutil.rmtree(config['output_root'])
        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)
        print(f"✅ 已删除临时文件")

if __name__ == "__main__":
    train_yolov8()
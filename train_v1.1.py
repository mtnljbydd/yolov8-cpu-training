import os
import shutil
import yaml
import sys
import time
import warnings
import torch
warnings.filterwarnings('ignore')

# ========================= 核心配置 =========================
# 读取YAML配置文件
CONFIG_PATH = "./config.yaml"  # 请确保该路径指向你的配置文件
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

# ========================= 自定义Trainer（单行进度条） =========================
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.engine.trainer import BaseTrainer

class CustomTrainer(BaseTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.epoch_start_time = 0
        self.batch_start_time = 0
        self.total_batches = 0
        
    def set_dataloader(self, dataloader):
        """设置数据加载器并记录总批次"""
        super().set_dataloader(dataloader)
        self.total_batches = len(self.train_loader) if hasattr(self, 'train_loader') else 0
    
    def train_epoch(self):
        """重写训练Epoch方法，添加单行进度条"""
        self.epoch_start_time = time.time()
        self.model.train()
        
        # 初始化进度条显示
        sys.stdout.write(f"\n      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n")
        sys.stdout.flush()
        
        for i, batch in enumerate(self.train_loader):
            self.batch_start_time = time.time()
            self.batch = batch
            
            # 执行批次训练
            self.train_step(batch)
            
            # 计算实时进度
            batch_elapsed = time.time() - self.batch_start_time
            epoch_elapsed = time.time() - self.epoch_start_time
            progress = (i + 1) / self.total_batches if self.total_batches > 0 else 1.0
            remaining_time = epoch_elapsed / (i + 1) * (self.total_batches - i - 1) if (i + 1) > 0 else 0
            
            # 格式化时间
            def format_time(seconds):
                mins, secs = divmod(int(seconds), 60)
                hrs, mins = divmod(mins, 60)
                if hrs > 0:
                    return f"{hrs}:{mins:02d}:{secs:02d}"
                return f"{mins:02d}:{secs:02d}"
            
            # 构建进度条
            bar_length = 20
            filled_length = int(bar_length * progress)
            bar = '━' * filled_length + '─' * (bar_length - filled_length)
            percent = int(progress * 100)
            
            # 获取真实的训练损失
            tloss = self.tloss if hasattr(self, 'tloss') else [0.0, 0.0, 0.0]
            box_loss = tloss[0] if len(tloss) >= 1 else 0.0
            cls_loss = tloss[1] if len(tloss) >= 2 else 0.0
            dfl_loss = tloss[2] if len(tloss) >= 3 else 0.0
            
            # 获取批次中的实例数量
            instances = len(batch[0]) if len(batch) > 0 else 0
            
            # 构建单行输出（\r 回到行首覆盖）
            line = (
                f"\r       {self.epoch+1}/{self.epochs}         0G      {box_loss:.3f}      {cls_loss:.3f}      {dfl_loss:.3f}         {instances}        {self.args.imgsz}: "
                f"{percent}% {bar} {i+1}/{self.total_batches} [{format_time(epoch_elapsed)}<{format_time(remaining_time)}, {1/batch_elapsed:.1f}it/s]"
            )
            
            # 打印单行进度条
            sys.stdout.write(line)
            sys.stdout.flush()
        
        # Epoch结束：换行
        sys.stdout.write('\n')
        return self.tloss

# ========================= 核心训练逻辑 =========================
def train_yolov8():
    """基于YAML配置的YOLOv8训练函数（仅CPU）"""
    # 1. 提取配置参数
    dataset_cfg = CONFIG['dataset']
    export_cfg = CONFIG['export']
    model_naming_cfg = CONFIG['model_naming']
    paths_cfg = CONFIG['paths']
    training_cfg = CONFIG['training']
    validation_cfg = CONFIG['validation']
    
    # 2. 打印配置信息（验证读取是否正确）
    print("="*50)
    print("📌 训练配置（仅CPU）")
    print(f"   模型：{training_cfg['model']} | 批次：{training_cfg['batch']} | 轮数：{training_cfg['epochs']}")
    print(f"   图片尺寸：{training_cfg['imgsz']} | 学习率：{training_cfg['learning_rate']}")
    print(f"   数据集类别数：{len(dataset_cfg['class_dict'])}")
    print("="*50)
    
    # 3. 生成临时data.yaml（YOLO训练需要）
    temp_data_yaml = "./config/temp_data.yaml"
    os.makedirs(os.path.dirname(temp_data_yaml), exist_ok=True)
    temp_data = {
        "train": paths_cfg['train'],
        "val": paths_cfg['val'],
        "nc": len(dataset_cfg['class_dict']),
        "names": list(dataset_cfg['class_dict'].values())  # 使用英文标识作为类别名
    }
    with open(temp_data_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(temp_data, f, indent=2, allow_unicode=True)
    print(f"✅ 生成临时data.yaml：{temp_data_yaml}")
    
    # 4. 加载预训练模型
    model = YOLO(training_cfg['model'])
    print(f"✅ 加载模型：{training_cfg['model']}")
    
    # 5. 构建训练参数（完全从YAML读取）
    train_args = {
        "data": temp_data_yaml,
        "epochs": training_cfg['epochs'],
        "batch": training_cfg['batch'],
        "imgsz": training_cfg['imgsz'],
        "device": training_cfg['device'],
        "patience": training_cfg['patience'],
        "save": True,
        "save_period": training_cfg['save_period'],
        "lr0": training_cfg['learning_rate'],
        "weight_decay": training_cfg['weight_decay'],
        "momentum": training_cfg['momentum'],
        "warmup_epochs": training_cfg['warmup_epochs'],
        "val": True,
        "cache": "ram",  # 16GB内存建议ram（参考YAML备注）
        "verbose": False,
        "project": training_cfg['project'],
        "name": training_cfg['name'],
        "exist_ok": training_cfg['exist_ok'],
        "plots": validation_cfg['plots'],
        "save_json": validation_cfg['save_json'],
        "workers": 8,  # 参考YAML备注：16GB内存建议8
        "single_cls": False,
        # 验证参数
        "conf": validation_cfg['conf'],
        "iou": validation_cfg['iou']
    }
    
    # 6. 开始训练（使用自定义Trainer，单行进度条）
    print("\n🚀 开始训练（仅CPU）...")
    model.trainer = CustomTrainer(overrides=train_args)
    try:
        results = model.train(**train_args)
        
        # 打印验证结果
        val_metrics = results.metrics
        print("\n" + "="*50)
        print("📊 验证结果")
        print(f"   验证集box_loss：{val_metrics.get('val/box_loss', 0.0):.4f}")
        print(f"   验证集cls_loss：{val_metrics.get('val/cls_loss', 0.0):.4f}")
        print(f"   mAP50：{val_metrics.get('metrics/mAP50(B)', 0.0):.4f}")
        print(f"   mAP50-95：{val_metrics.get('metrics/mAP50-95(B)', 0.0):.4f}")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 训练出错：{str(e)}")
        return
    
    # 7. 重命名模型（按配置中的命名规则）
    model_name = f"{model_naming_cfg['model_name']}_{model_naming_cfg['model_version']}"
    original_exp_dir = os.path.join(training_cfg['project'], training_cfg['name'])
    final_exp_dir = os.path.join(training_cfg['project'], model_name)
    
    if os.path.exists(original_exp_dir) and not os.path.exists(final_exp_dir):
        os.rename(original_exp_dir, final_exp_dir)
        print(f"\n✅ 模型重命名：{original_exp_dir} → {final_exp_dir}")
    
    # 8. 导出模型（按配置）
    export_path = paths_cfg['export_save_path'].format(
        root_path=paths_cfg['root_path'],
        model_name=model_naming_cfg['model_name'],
        model_version=model_naming_cfg['model_version']
    )
    best_model_path = os.path.join(final_exp_dir, "weights", "best.pt")
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        # 导出为OpenVINO格式（CPU加速）
        best_model.export(
            format='openvino',
            imgsz=export_cfg['imgsz'],
            batch=export_cfg['batch'],
            device=export_cfg['device'],
            save_dir=export_path
        )
        print(f"✅ 模型导出完成：{export_path}")
    
    # 9. 清理临时文件（按配置）
    if dataset_cfg['delete_temp_files'] and os.path.exists(temp_data_yaml):
        os.remove(temp_data_yaml)
        print(f"✅ 清理临时文件：{temp_data_yaml}")
    
    print("\n🎉 训练流程全部完成！")
    print(f"📌 最终模型路径：{final_exp_dir}")
    print(f"📌 导出模型路径：{export_path}")

# ========================= 执行训练 =========================
if __name__ == "__main__":
    # 校验CPU环境
    if torch.cuda.is_available():
        print("⚠️  检测到GPU，但配置指定仅CPU训练，将强制使用CPU")
    train_yolov8()
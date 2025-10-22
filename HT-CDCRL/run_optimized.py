import torch
import random
import numpy as np
import os
from pytorch_lightning import Trainer
import swanlab as wandb


# 设置临时目录到用户目录，避免 /tmp 空间不足
os.environ['TMPDIR'] = '/home/'
os.makedirs('/home/', exist_ok=True)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging

# 假设您的代码库在Python的搜索路径中
# 如果不在，您可能需要使用 sys.path.append()
from src.models.MAPPO_optimized import OptimizedMAPPO as MAPPO
from src.datamodules.MAPPODatamodule import MAPPODatamodule
from src.utils.logging_manager import LoggingManager

# ============== SwanLab 集成（Lightning 回调 + 运行初始化） ==============
from pytorch_lightning.callbacks import Callback

class SwanLabLoggerCallback(Callback):
    """将训练/验证/测试阶段的指标同步到 SwanLab。"""

    def on_train_start(self, trainer, pl_module):
        # 记录模型规模
        try:
            total_params = sum(p.numel() for p in pl_module.parameters())
            trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            wandb.log({
                "params/total": int(total_params),
                "params/trainable": int(trainable_params)
            })
        except Exception:
            pass

    def _to_scalar(self, value):
        """尽量把各种类型(张量、numpy、torchmetrics)转为float。失败则返回None。"""
        try:
            # torchmetrics.Metric
            if hasattr(value, "compute") and callable(getattr(value, "compute")):
                value = value.compute()
            # torch.Tensor
            if hasattr(value, "detach"):
                value = value.detach().cpu()
            # numpy scalar
            import numpy as _np  # 局部导入，避免顶层依赖
            if hasattr(value, "item"):
                return float(value.item())
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, _np.ndarray) and value.size == 1:
                return float(value.reshape(()))
        except Exception:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _log_metrics(self, trainer, prefixes):
        try:
            metrics = {}
            for k, v in trainer.callback_metrics.items():
                if isinstance(k, str) and (not prefixes or any(k.startswith(p) for p in prefixes)):
                    scalar = self._to_scalar(v)
                    if scalar is not None:
                        metrics[k] = scalar
            if metrics:
                # 让 SwanLab 自动步进
                wandb.log(metrics)
        except Exception:
            pass

    # 训练/验证/测试在 epoch 结束时都打点，确保曲线出现
    def on_train_epoch_end(self, trainer, pl_module):
        self._log_metrics(trainer, ("train/",))

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_metrics(trainer, ("val/",))

    def on_test_epoch_end(self, trainer, pl_module):
        self._log_metrics(trainer, ("test/",))

    # 训练每个batch结束也打点，确保曲线实时更新
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log_metrics(trainer, ("train/",))

def get_optimized_config():
    """
    基于实验结果优化的配置
    主要改进：
    1. 增加模型容量（hidden_dim）
    2. 优化学习率调度
    3. 改进正则化策略
    4. 增强奖励信号
    5. 动态候选集策略
    """
    
    # 使用更保守的候选集大小以提升早期Hit@1/5
    candi_merchant_num = 15
    
    config = {
        "project_name": "poi_recommender_optimized",
        "experiment_name": f"optimized_candi_{candi_merchant_num}",
        "seed": 5521,

        # Trainer 配置 - 优化训练策略
        "trainer": {
            "devices": 1,
            "max_epochs": 200,  # 增加训练轮数，配合更强的早停
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "log_every_n_steps": 25,  # 更频繁的日志记录
            "gradient_clip_val": 1.0,  # 适度放松梯度裁剪
            "precision": "16-mixed",
            "accumulate_grad_batches": 2,  # 梯度累积，等效增加batch size
            "check_val_every_n_epoch": 2,  # 每2个epoch验证一次，节省时间
        },

        # Model (MAPPO) 配置 - 重点优化
        "model": {
            "data_dir": "data/NYC",
            "obs_dim": 256,  # 🔥 增加观察维度，提升表示能力
            "topk": [1, 5, 10, 15],
            "batch_size": 48,  # 提升稳定性
            "committee_weight": 0.8,  # 更强调候选质量
            "lr": 5e-4,  # 降低学习率，减小更新噪声
            "weight_decay": 5e-5,  # 略减正则避免欠拟合
            "update_steps": 8,  # 稍降更新强度，提升稳定性
            "candi_merchant_num": candi_merchant_num,
            "cat_num": 399,
            
            # 🔥 优化奖励函数参数
            "reward_alpha": 18.0,   # 更强地激励Hit@1
            "reward_beta": 4.0,     # 增强ranking/margin
            "reward_gamma": 0.25,   # 稍降多样性
            
            # 🔥 优化PPO参数
            "clip_ratio": 0.12,     # 略收紧，稳定优化
            "seed": 42,
            
            # 🔥 排序融合系数
            "ranking_alpha": 0.55,
        },

        # Datamodule 配置 - 数据效率优化
        "datamodule": {
            "data_dir": "data/NYC",
            "batch_size": 48,       # 与model保持一致
            "test_batch_size": 128, # 🔥 测试时可以用更大batch
            "num_workers": 4,       # 提高数据吞吐（若环境允许）
            "seed": 42,
        },

        # 统一早停配置（允许外部覆盖）：例如 5 表示验证 5 次无提升即停止
        "early_stopping_patience": 5,
    }
    
    return config

def get_optimized_callbacks(config):
    """优化的回调函数配置"""
    callbacks = []
    
    # 🔥 改进的模型检查点策略
    checkpoint_callback = ModelCheckpoint(
        dirpath="/data/liuy/experiments/checkpoints/optimized/",
        filename=f"{config['experiment_name']}-{{epoch:02d}}-{{val/Hit@10:.3f}}-{{val/NDCG@20:.3f}}",
        save_top_k=3,  # 保存top3模型
        save_last=True,
        verbose=True,
        monitor="val/Hit@10",  # 主要监控Hit@10
        mode="max",
        auto_insert_metric_name=False,
        every_n_epochs=5,  # 每5个epoch保存一次，减少I/O
    )
    callbacks.append(checkpoint_callback)
    
    # 🔥 改进的早停策略 - 更耐心但更精确
    early_stopping_callback = EarlyStopping(
        monitor="val/Hit@5",  # 监控Hit@5，较为敏感的指标
        patience=config.get("early_stopping_patience", 25),  # 从配置读取，默认25
        mode="max", 
        verbose=True, 
        strict=True, 
        check_finite=True,
        min_delta=0.005,      # 降低最小改进阈值
        stopping_threshold=0.6,  # 如果Hit@5达到60%就可以考虑停止
    )
    callbacks.append(early_stopping_callback)
    
    # 🔥 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # 🔥 随机权重平均 - 提升最终性能
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-4, swa_epoch_start=150)
    callbacks.append(swa_callback)
    
    # SwanLab 指标同步
    callbacks.append(SwanLabLoggerCallback())
    
    return callbacks

def main():
    """优化主函数"""
    print("🚀 启动参数优化实验")
    
    # 获取优化配置
    config = get_optimized_config()
    
    # 🔥 初始化规范的日志管理器 - 保存到 /data 分区
    logging_manager = LoggingManager(
        base_dir="/data/liuy/experiments/logs",
        dataset_name="NYC",  # 从config中获取
        model_name="OptimizedMAPPO",
        experiment_name=config.get("experiment_name", "default_exp")
    )
    
    # 保存配置信息
    logging_manager.save_config(config)
    print(f"📁 日志目录: {logging_manager.run_dir}")
    
    # 初始化 SwanLab 运行
    wandb_run = wandb.init(
        project=config.get("project_name", "poi_recommender"),
        name=config.get("experiment_name", None),
        config=config,
    )
    
    # 设置随机种子
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 🔥 优化CUDA设置
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 实例化组件
    print("📊 实例化数据模块...")
    datamodule = MAPPODatamodule(**config["datamodule"])

    print("🧠 实例化优化模型...")
    model = MAPPO(**config["model"], logging_manager=logging_manager)
    
    # 🔥 模型优化后打印参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # Logger - 保存到 /data 分区
    logger = TensorBoardLogger("/data/liuy/experiments/tb_logs", name=config["experiment_name"])

    # 获取优化的回调函数
    callbacks = get_optimized_callbacks(config)

    # 实例化并启动Trainer
    print("⚡ 实例化优化训练器...")
    trainer = Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=callbacks,
        #  添加训练优化选项
        enable_model_summary=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
        detect_anomaly=False,  # 生产环境关闭异常检测
    )

    # 开始训练
    print("🎯 开始优化训练!")
    print(f"配置概要:")
    print(f"  候选集大小: {config['model']['candi_merchant_num']}")
    print(f"  观察维度: {config['model']['obs_dim']}")
    print(f"  批次大小: {config['model']['batch_size']}")
    print(f"  学习率: {config['model']['lr']}")
    print(f"  更新步数: {config['model']['update_steps']}")
    print(f"  奖励参数: α={config['model']['reward_alpha']}, β={config['model']['reward_beta']}, γ={config['model']['reward_gamma']}")
    
    trainer.fit(model=model, datamodule=datamodule)

    print("🧪 开始测试!")
    test_results = trainer.test(model=model, datamodule=datamodule)
    
    print("✅ 优化实验完成!")
    if test_results:
        print("最终测试结果:")
        for metric, value in test_results[0].items():
            print(f"  {metric}: {value:.4f}")
            try:
                wandb.log({f"test/{metric}": float(value)})
            except Exception:
                pass
    
    # 结束 SwanLab 会话
    try:
        wandb.finish()
    except Exception:
        pass

    return test_results

if __name__ == "__main__":
    # 确保数据目录存在
    if not os.path.exists("data/NYC"):
        print("❌ 错误：数据目录 'data/NYC' 不存在。请将数据文件放在该目录下。")
        exit()

    if not os.path.exists("checkpoints/optimized"):
        os.makedirs("checkpoints/optimized", exist_ok=True)
    
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)

    # 运行优化实验
    results = main()



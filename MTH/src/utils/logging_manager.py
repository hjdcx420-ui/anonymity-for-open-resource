"""
🔥 规范的日志管理系统
按照数据集和模型保存规范，统一管理训练和推理过程中的日志记录
"""

import os
import json
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
import logging

class LoggingManager:
    """统一的日志管理器"""
    
    def __init__(self, 
                 base_dir: str = "/data/liuy/experiments/logs",
                 dataset_name: str = "London",
                 model_name: str = "OptimizedMAPPO",
                 experiment_name: str = "default_exp",
                 run_id: Optional[str] = None):
        """
        初始化日志管理器
        
        Args:
            base_dir: 基础日志目录
            dataset_name: 数据集名称 (如: London, CAL, NYC)
            model_name: 模型名称 (如: OptimizedMAPPO, MAPPO)
            experiment_name: 实验名称
            run_id: 运行ID，如果为None则自动生成
        """
        self.base_dir = Path(base_dir)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.experiment_name = experiment_name
        
        # 生成运行ID
        if run_id is None:
            self.run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        else:
            self.run_id = run_id
            
        # 构建目录结构
        self.run_dir = self.base_dir / dataset_name / model_name / experiment_name / self.run_id
        
        # 创建子目录
        self.csv_dir = self.run_dir / "csv"
        self.pred_distributions_dir = self.run_dir / "pred_distributions"
        self.user_embeddings_dir = self.run_dir / "user_embeddings"
        self.batch_rewards_dir = self.run_dir / "batch_rewards"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.config_dir = self.run_dir / "config"
        
        # 创建所有目录
        for dir_path in [self.csv_dir, self.pred_distributions_dir, 
                        self.user_embeddings_dir, self.batch_rewards_dir,
                        self.checkpoints_dir, self.config_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # 初始化CSV文件
        self._init_csv_files()
        
        # 设置日志记录
        self._setup_logging()
        
    def _init_csv_files(self):
        """初始化CSV文件"""
        # 预测分布CSV
        self.preds_val_csv = self.csv_dir / "preds_val.csv"
        self.preds_test_csv = self.csv_dir / "preds_test.csv"
        
        # 用户表征CSV
        self.user_embeddings_csv = self.csv_dir / "user_embeddings.csv"
        
        # 激励CSV
        self.rewards_csv = self.csv_dir / "rewards.csv"
        
        # 训练指标CSV
        self.metrics_csv = self.csv_dir / "training_metrics.csv"
        
        # 初始化CSV文件头
        self._init_csv_headers()
        
    def _init_csv_headers(self):
        """初始化CSV文件头"""
        # 预测分布CSV头
        pred_headers = [
            "dataset", "model", "experiment", "run_id", "epoch", "batch_idx", 
            "split", "user_id", "gt_poi", "pred_scores", "candidate_ids", 
            "gt_position", "timestamp", "pid"
        ]
        
        # 用户表征CSV头
        embedding_headers = [
            "dataset", "model", "experiment", "run_id", "epoch", "user_id", 
            "embedding_vector", "timestamp", "pid"
        ]
        
        # 激励CSV头
        reward_headers = [
            "dataset", "model", "experiment", "run_id", "epoch", "batch_idx",
            "user_rewards", "merchant_rewards", "total_rewards", "timestamp", "pid"
        ]
        
        # 训练指标CSV头
        metrics_headers = [
            "dataset", "model", "experiment", "run_id", "epoch", "phase",
            "loss", "reward", "ndcg_5", "ndcg_10", "ndcg_15", "hit_5", "hit_10", "hit_15",
            "timestamp", "pid"
        ]
        
        # 创建CSV文件（如果不存在）
        for csv_file, headers in [
            (self.preds_val_csv, pred_headers),
            (self.preds_test_csv, pred_headers),
            (self.user_embeddings_csv, embedding_headers),
            (self.rewards_csv, reward_headers),
            (self.metrics_csv, metrics_headers)
        ]:
            if not csv_file.exists():
                pd.DataFrame(columns=headers).to_csv(csv_file, index=False)
                
    def _setup_logging(self):
        """设置日志记录"""
        log_file = self.run_dir / "training.log"
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"LoggingManager initialized: {self.run_dir}")
        
    def save_config(self, config: Dict[str, Any]):
        """保存配置信息"""
        config_file = self.config_dir / "config.json"
        
        # 添加元数据
        config_with_meta = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "experiment": self.experiment_name,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": config
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_with_meta, f, indent=2)
            
        self.logger.info(f"Config saved to {config_file}")
        
    def save_prediction_distributions(self, 
                                    preds: torch.Tensor,
                                    candidate_ids: torch.Tensor,
                                    gt_positions: torch.Tensor,
                                    user_ids: torch.Tensor,
                                    epoch: int,
                                    batch_idx: int,
                                    split: str = "val"):
        """保存预测分布"""
        # 保存.pt文件
        pt_file = self.pred_distributions_dir / f"epoch_{epoch:04d}_batch_{batch_idx:05d}_{split}.pt"
        
        torch.save({
            "preds": preds.cpu(),
            "candidate_ids": candidate_ids.cpu(),
            "gt_positions": gt_positions.cpu(),
            "user_ids": user_ids.cpu(),
            "epoch": epoch,
            "batch_idx": batch_idx,
            "split": split,
            "timestamp": datetime.now().isoformat()
        }, pt_file)
        
        # 保存到CSV
        csv_file = self.preds_val_csv if split == "val" else self.preds_test_csv
        
        # 准备数据
        data = []
        for i in range(len(preds)):
            data.append({
                "dataset": self.dataset_name,
                "model": self.model_name,
                "experiment": self.experiment_name,
                "run_id": self.run_id,
                "epoch": epoch,
                "batch_idx": batch_idx,
                "split": split,
                "user_id": user_ids[i].item(),
                "gt_poi": candidate_ids[i, gt_positions[i]].item() if gt_positions[i] >= 0 else -1,
                "pred_scores": preds[i].cpu().tolist(),
                "candidate_ids": candidate_ids[i].cpu().tolist(),
                "gt_position": gt_positions[i].item(),
                "timestamp": datetime.now().isoformat(),
                "pid": os.getpid()
            })
        
        # 追加到CSV
        df = pd.DataFrame(data)
        df.to_csv(csv_file, mode='a', header=False, index=False)
        
        self.logger.info(f"Prediction distributions saved: {pt_file}")
        
    def save_user_embeddings(self, 
                           user_embeddings: torch.Tensor,
                           user_ids: torch.Tensor,
                           epoch: int):
        """保存用户表征"""
        # 保存.pt文件
        pt_file = self.user_embeddings_dir / f"epoch_{epoch:04d}.pt"
        
        torch.save({
            "user_embeddings": user_embeddings.cpu(),
            "user_ids": user_ids.cpu(),
            "epoch": epoch,
            "timestamp": datetime.now().isoformat()
        }, pt_file)
        
        # 保存到CSV
        data = []
        for i in range(len(user_embeddings)):
            data.append({
                "dataset": self.dataset_name,
                "model": self.model_name,
                "experiment": self.experiment_name,
                "run_id": self.run_id,
                "epoch": epoch,
                "user_id": user_ids[i].item(),
                "embedding_vector": user_embeddings[i].cpu().tolist(),
                "timestamp": datetime.now().isoformat(),
                "pid": os.getpid()
            })
        
        df = pd.DataFrame(data)
        df.to_csv(self.user_embeddings_csv, mode='a', header=False, index=False)
        
        self.logger.info(f"User embeddings saved: {pt_file}")
        
    def save_batch_rewards(self, 
                         user_rewards: torch.Tensor,
                         merchant_rewards: torch.Tensor,
                         epoch: int,
                         batch_idx: int):
        """保存batch激励"""
        # 保存.pt文件
        pt_file = self.batch_rewards_dir / f"epoch_{epoch:04d}_batch_{batch_idx:05d}.pt"
        
        torch.save({
            "user_rewards": user_rewards.cpu(),
            "merchant_rewards": merchant_rewards.cpu(),
            "total_rewards": user_rewards + merchant_rewards,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "timestamp": datetime.now().isoformat()
        }, pt_file)
        
        # 保存到CSV
        data = []
        for i in range(len(user_rewards)):
            data.append({
                "dataset": self.dataset_name,
                "model": self.model_name,
                "experiment": self.experiment_name,
                "run_id": self.run_id,
                "epoch": epoch,
                "batch_idx": batch_idx,
                "user_rewards": user_rewards[i].item(),
                "merchant_rewards": merchant_rewards[i].item(),
                "total_rewards": (user_rewards[i] + merchant_rewards[i]).item(),
                "timestamp": datetime.now().isoformat(),
                "pid": os.getpid()
            })
        
        df = pd.DataFrame(data)
        df.to_csv(self.rewards_csv, mode='a', header=False, index=False)
        
        self.logger.info(f"Batch rewards saved: {pt_file}")
        
    def save_training_metrics(self, 
                            epoch: int,
                            phase: str,
                            metrics: Dict[str, float]):
        """保存训练指标"""
        data = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "experiment": self.experiment_name,
            "run_id": self.run_id,
            "epoch": epoch,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid()
        }
        
        # 添加指标
        data.update(metrics)
        
        df = pd.DataFrame([data])
        df.to_csv(self.metrics_csv, mode='a', header=False, index=False)
        
        self.logger.info(f"Training metrics saved for epoch {epoch}, phase {phase}")
        
    def save_checkpoint(self, 
                       model_state_dict: Dict[str, torch.Tensor],
                       optimizer_state_dict: Dict[str, Any],
                       epoch: int,
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "metrics": metrics,
            "dataset": self.dataset_name,
            "model": self.model_name,
            "experiment": self.experiment_name,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # 保存检查点
        checkpoint_file = self.checkpoints_dir / f"epoch_{epoch:04d}.ckpt"
        torch.save(checkpoint, checkpoint_file)
        
        # 保存最佳模型
        if is_best:
            best_file = self.checkpoints_dir / "best_model.ckpt"
            torch.save(checkpoint, best_file)
            
        # 保存最新模型
        latest_file = self.checkpoints_dir / "latest.ckpt"
        torch.save(checkpoint, latest_file)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_file}")
        
    def get_run_info(self) -> Dict[str, str]:
        """获取运行信息"""
        return {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "experiment": self.experiment_name,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir)
        }
        
    def cleanup_old_runs(self, keep_last_n: int = 5):
        """清理旧的运行记录，只保留最近的N个"""
        experiment_dir = self.run_dir.parent
        
        # 获取所有运行目录
        run_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        run_dirs.sort(key=lambda x: x.name, reverse=True)
        
        # 删除旧的运行目录
        for old_run_dir in run_dirs[keep_last_n:]:
            import shutil
            shutil.rmtree(old_run_dir)
            self.logger.info(f"Cleaned up old run: {old_run_dir}")
            
    def __str__(self):
        return f"LoggingManager(dataset={self.dataset_name}, model={self.model_name}, run_id={self.run_id})"

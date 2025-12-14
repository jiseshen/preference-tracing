"""
检查点管理器 - 管理用户追踪的进度和断点续跑
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Set

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, run_id: str):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
            run_id: 运行标识符
        """
        self.checkpoint_dir = checkpoint_dir
        self.run_id = run_id
        self.checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{run_id}.json")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 加载或初始化检查点
        self.checkpoint = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict:
        """加载检查点文件"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                'run_id': self.run_id,
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'completed_users': [],
                'failed_users': [],
                'user_status': {}
            }
    
    def _save_checkpoint(self):
        """保存检查点到文件"""
        self.checkpoint['last_updated'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.checkpoint, f, indent=2, ensure_ascii=False)
    
    def is_user_completed(self, user_id: str) -> bool:
        """检查用户是否已完成"""
        return user_id in self.checkpoint['completed_users']
    
    def is_user_failed(self, user_id: str) -> bool:
        """检查用户是否失败"""
        return user_id in self.checkpoint['failed_users']
    
    def mark_user_started(self, user_id: str):
        """标记用户开始处理"""
        self.checkpoint['user_status'][user_id] = {
            'status': 'in_progress',
            'started_at': datetime.now().isoformat()
        }
        self._save_checkpoint()
    
    def mark_user_completed(self, user_id: str, turns_completed: int = None):
        """标记用户完成"""
        if user_id not in self.checkpoint['completed_users']:
            self.checkpoint['completed_users'].append(user_id)
        
        self.checkpoint['user_status'][user_id] = {
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'turns_completed': turns_completed
        }
        self._save_checkpoint()
    
    def mark_user_failed(self, user_id: str, error: str = None):
        """标记用户失败"""
        if user_id not in self.checkpoint['failed_users']:
            self.checkpoint['failed_users'].append(user_id)
        
        self.checkpoint['user_status'][user_id] = {
            'status': 'failed',
            'failed_at': datetime.now().isoformat(),
            'error': error
        }
        self._save_checkpoint()
    
    def get_completed_users(self) -> List[str]:
        """获取已完成的用户列表"""
        return self.checkpoint['completed_users']
    
    def get_failed_users(self) -> List[str]:
        """获取失败的用户列表"""
        return self.checkpoint['failed_users']
    
    def get_pending_users(self, all_users: List[str]) -> List[str]:
        """获取待处理的用户列表"""
        completed = set(self.checkpoint['completed_users'])
        return [u for u in all_users if u not in completed]
    
    def get_progress_summary(self) -> Dict:
        """获取进度摘要"""
        return {
            'run_id': self.run_id,
            'completed': len(self.checkpoint['completed_users']),
            'failed': len(self.checkpoint['failed_users']),
            'last_updated': self.checkpoint['last_updated']
        }
    
    def reset(self):
        """重置检查点（清空所有进度）"""
        self.checkpoint = {
            'run_id': self.run_id,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'completed_users': [],
            'failed_users': [],
            'user_status': {}
        }
        self._save_checkpoint()

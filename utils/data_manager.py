import os
import json
from datetime import datetime
from typing import Dict, List, Any
from contextlib import redirect_stdout
import io
import sys

class DataManager:
    def __init__(self, output_dir: str, run_id: str, save_logs: bool = True):
        self.output_dir = output_dir
        self.run_id = run_id
        self.save_logs = save_logs
        
        self.run_dir = os.path.join(output_dir, run_id)
        self.logs_dir = os.path.join(self.run_dir, 'logs')
        self.metrics_dir = os.path.join(self.run_dir, 'metrics')
        self.traces_dir = os.path.join(self.run_dir, 'traces')
        
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.traces_dir, exist_ok=True)
        
        # 初始化数据存储
        self.all_metrics = {
            'run_id': run_id,
            'created_at': datetime.now().isoformat(),
            'users': {}
        }
        
        self.current_user_log = None
        self.current_user_id = None
    
    def start_user_logging(self, user_id: str):
        """开始记录某个用户的日志"""
        self.current_user_id = user_id
        if self.save_logs:
            self.current_user_log = []
    
    def log(self, message: str):
        """记录日志消息"""
        if self.save_logs and self.current_user_log is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.current_user_log.append(f"[{timestamp}] {message}")
    
    def save_user_log(self):
        """保存当前用户的日志到文件"""
        if self.save_logs and self.current_user_log is not None and self.current_user_id:
            log_file = os.path.join(self.logs_dir, f"user_{self.current_user_id}.log")
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.current_user_log))
            self.current_user_log = None
            self.current_user_id = None
    
    def save_turn_metrics(self, user_id: str, turn_idx: int, metrics: Dict[str, Any]):
        """
        保存单轮的评估指标
        
        Args:
            user_id: 用户ID
            turn_idx: 轮次索引
            metrics: 指标字典，包括：
                - generation_scores: 生成分数
                - prediction_correct: 预测是否正确
                - ess: 有效样本大小
                - diversity: 多样性
                - weights: 假设权重
                - etc.
        """
        if user_id not in self.all_metrics['users']:
            self.all_metrics['users'][user_id] = {
                'turns': [],
                'final_alignment': None
            }
        
        turn_data = {
            'turn_idx': turn_idx,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.all_metrics['users'][user_id]['turns'].append(turn_data)
        
        # 实时保存（防止中断丢失数据）
        self._save_metrics()
    
    def save_final_alignment(self, user_id: str, alignment_score: float, 
                            survey_comparison: Dict = None):
        """
        保存最终的对齐分数
        
        Args:
            user_id: 用户ID
            alignment_score: 对齐分数
            survey_comparison: 与survey的比较结果
        """
        if user_id not in self.all_metrics['users']:
            self.all_metrics['users'][user_id] = {'turns': []}
        
        self.all_metrics['users'][user_id]['final_alignment'] = {
            'score': alignment_score,
            'timestamp': datetime.now().isoformat(),
            'survey_comparison': survey_comparison
        }
        
        self._save_metrics()
    
    def save_user_trace(self, user_id: str, trace_data: Dict):
        """
        保存用户的完整追踪数据
        
        Args:
            user_id: 用户ID
            trace_data: 追踪数据（包括所有假设、历史等）
        """
        trace_file = os.path.join(self.traces_dir, f"trace_user_{user_id}.json")
        with open(trace_file, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
    
    def _save_metrics(self):
        """保存所有指标到文件"""
        metrics_file = os.path.join(self.metrics_dir, 'all_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_metrics, f, indent=2, ensure_ascii=False)
    
    def get_user_metrics(self, user_id: str) -> Dict:
        """获取某个用户的所有指标"""
        return self.all_metrics['users'].get(user_id, {})
    
    def get_summary_statistics(self) -> Dict:
        """获取汇总统计信息"""
        users = self.all_metrics['users']
        
        if not users:
            return {'total_users': 0}
        
        total_turns = sum(len(u['turns']) for u in users.values())
        completed_users = sum(1 for u in users.values() if u.get('final_alignment'))
        
        # 计算平均对齐分数
        alignment_scores = [
            u['final_alignment']['score'] 
            for u in users.values() 
            if u.get('final_alignment')
        ]
        avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
        
        # 计算平均预测准确率
        all_predictions = []
        for user_data in users.values():
            for turn in user_data['turns']:
                if 'prediction_correct' in turn:
                    all_predictions.append(turn['prediction_correct'])
        
        avg_accuracy = sum(all_predictions) / len(all_predictions) if all_predictions else 0
        
        return {
            'total_users': len(users),
            'completed_users': completed_users,
            'total_turns': total_turns,
            'average_alignment_score': avg_alignment,
            'average_prediction_accuracy': avg_accuracy,
            'alignment_scores': alignment_scores
        }
    
    def export_summary(self) -> str:
        summary = self.get_summary_statistics()
        summary_file = os.path.join(self.run_dir, 'summary.json')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary_file


class LogCapture:
    """上下文管理器，用于捕获打印输出到数据管理器"""
    def __init__(self, data_manager: DataManager, also_print: bool = True):
        self.data_manager = data_manager
        self.also_print = also_print
        self.buffer = io.StringIO()
    
    def __enter__(self):
        if not self.also_print:
            self._old_stdout = sys.stdout
            sys.stdout = self.buffer
        return self
    
    def __exit__(self, *args):
        if not self.also_print:
            sys.stdout = self._old_stdout
            output = self.buffer.getvalue()
            if output:
                self.data_manager.log(output)

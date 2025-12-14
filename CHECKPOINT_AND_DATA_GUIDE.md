# 检查点和数据管理使用指南

本指南介绍如何使用新增的检查点管理和数据持久化功能。

## 🎯 核心功能

### 1. 断点续跑（Checkpoint Resume）
- ✅ 自动记录每个用户的处理状态
- ✅ 程序中断后可从上次位置继续
- ✅ 避免重复处理已完成的用户

### 2. 完整数据保存
- ✅ 所有评估指标自动保存（不受 `--print` 参数影响）
- ✅ 按用户组织的详细日志文件
- ✅ 结构化的指标数据（JSON格式）

### 3. 日志管理
- ✅ 每个用户的完整处理日志
- ✅ 可选择同时在终端显示和保存到文件
- ✅ 时间戳和详细的步骤记录

---

## 📁 输出目录结构

运行后会在输出目录创建以下结构：

```
preference_results/
├── checkpoints/
│   └── checkpoint_<run_id>.json          # 检查点状态
├── logs/
│   └── <run_id>/
│       ├── user_001.log                  # 用户001的处理日志
│       ├── user_002.log
│       └── ...
├── metrics/
│   └── <run_id>/
│       └── all_metrics.json              # 所有用户的评估指标
├── traces/
│   └── <run_id>/
│       ├── trace_user_001.json           # 用户001的完整追踪数据
│       ├── trace_user_002.json
│       └── ...
├── preference_tracing_results_<run_id>.json  # 主结果文件
└── summary_<run_id>.json                     # 汇总统计
```

---

## 🚀 使用方法

### 基本运行

```bash
# 运行追踪（自动启用检查点）
python run_prism_pipeline.py \
    --stage trace \
    --n-users 50 \
    --run-id my_experiment

# 启用终端日志显示 + 文件保存
python run_prism_pipeline.py \
    --stage trace \
    --n-users 50 \
    --run-id my_experiment \
    --print
```

### 断点续跑

```bash
# 如果程序中断，直接重新运行相同命令即可自动恢复
python run_prism_pipeline.py \
    --stage trace \
    --n-users 50 \
    --run-id my_experiment \
    --resume

# 输出会显示：
# 📌 Found existing checkpoint for run 'my_experiment':
#    - Completed: 23 users
#    - Failed: 1 users
#    - Last updated: 2025-10-26T10:30:45
#
# 📊 Processing Status:
#    Total users: 50
#    Completed: 23
#    Remaining: 26
#    Failed: 1
```

### 重置检查点

```bash
# 如果想从头开始重新运行
python run_prism_pipeline.py \
    --stage trace \
    --n-users 50 \
    --run-id my_experiment \
    --reset-checkpoint
```

### 使用配置文件

在 `configs.yaml` 中定义配置：

```yaml
production_run:
  n_users: 100
  n_hypotheses: 4
  tracing_model: gpt-4o-mini
  eval_model: gpt-4o-mini
  output_dir: results/production
  run_id: prod_v1
  print: true
  seed: 42
```

然后运行：

```bash
python run_prism_pipeline.py --config production_run --stage trace --resume
```

---

## 📊 保存的数据详解

### 1. 检查点文件 (`checkpoint_<run_id>.json`)

记录处理进度：

```json
{
  "run_id": "my_experiment",
  "created_at": "2025-10-26T10:00:00",
  "last_updated": "2025-10-26T10:30:45",
  "completed_users": ["user_001", "user_002", "user_003"],
  "failed_users": ["user_042"],
  "user_status": {
    "user_001": {
      "status": "completed",
      "completed_at": "2025-10-26T10:05:12",
      "turns_completed": 5
    },
    "user_042": {
      "status": "failed",
      "failed_at": "2025-10-26T10:28:30",
      "error": "API rate limit exceeded"
    }
  }
}
```

### 2. 指标文件 (`metrics/<run_id>/all_metrics.json`)

**所有轮次的完整指标数据**（自动保存，不受 `--print` 影响）：

```json
{
  "run_id": "my_experiment",
  "created_at": "2025-10-26T10:00:00",
  "users": {
    "user_001": {
      "turns": [
        {
          "turn_idx": 0,
          "timestamp": "2025-10-26T10:05:00",
          "gen_score": 0.85,
          "prediction_correct": true,
          "predicted_idx": 1,
          "actual_idx": 1,
          "ess": 3.42,
          "text_diversity": 0.76,
          "resampled": false,
          "rejuvenated": false,
          "weights": [0.4, 0.3, 0.2, 0.1],
          "hypotheses": ["...", "...", "..."]
        },
        {
          "turn_idx": 1,
          "timestamp": "2025-10-26T10:05:15",
          "gen_score": 0.92,
          "prediction_correct": true,
          "ess": 2.18,
          "text_diversity": 0.42,
          "resampled": true,
          "rejuvenated": false
        }
      ],
      "final_alignment": {
        "score": 0.88,
        "timestamp": "2025-10-26T10:05:30",
        "survey_comparison": {...}
      }
    }
  }
}
```

**包含的指标**：
- ✅ `gen_score`: 每轮的生成分数
- ✅ `prediction_correct`: 预测是否正确（布尔值）
- ✅ `predicted_idx` / `actual_idx`: 预测和实际选择的候选索引
- ✅ `ess`: 有效样本大小（Effective Sample Size）
- ✅ `text_diversity`: 假设文本多样性
- ✅ `resampled` / `rejuvenated`: 是否触发重采样/重新生成
- ✅ `weights`: 每个假设的权重
- ✅ `hypotheses`: 当前假设文本（前3个）
- ✅ `final_alignment`: 最终与survey的对齐分数

### 3. 用户日志 (`logs/<run_id>/user_<id>.log`)

**详细的处理过程日志**（仅当使用 `--print` 时保存）：

```
[10:05:00] ======================================================================
[10:05:00] Tracing preferences for user: user_001
[10:05:00] ======================================================================
[10:05:02] === Initializing Hypotheses for User user_001 ===
[10:05:02]   1. The user values empathy and emotional understanding in responses
[10:05:02]   2. The user prefers concise, actionable advice over lengthy explanations
[10:05:02]   3. The user appreciates when the AI acknowledges uncertainty
[10:05:02]   4. The user wants responses that consider their personal context
[10:05:05] 
[10:05:05] --- Turn 0 ---
[10:05:05] User Message: I'm feeling overwhelmed with work. What should I do?...
[10:05:05] Chosen Response: I understand you're feeling overwhelmed...
[10:05:08] Gen Score: 0.85, Prediction: ✓
[10:05:10] 
[10:05:10] --- Turn 1 ---
[10:05:10] User Message: Can you help me prioritize my tasks?...
[10:05:12] Resampling (ESS: 1.82)
[10:05:15] Gen Score: 0.92, Prediction: ✓
[10:05:18] 
[10:05:18] Completed tracing for user user_001
[10:05:18] Total turns: 5
```

### 4. 追踪数据 (`traces/<run_id>/trace_user_<id>.json`)

每个用户的完整追踪数据（包括所有假设演化）：

```json
{
  "user_id": "user_001",
  "turn_results": [...],
  "final_profile": "This user values empathy and concise advice...",
  "detailed_hypotheses": [
    {
      "texts": [...],
      "weights": [...],
      "contexts": [...],
      "perceptions": [...]
    }
  ]
}
```

### 5. 汇总统计 (`summary_<run_id>.json`)

整体统计信息：

```json
{
  "total_users": 50,
  "completed_users": 49,
  "total_turns": 245,
  "average_alignment_score": 0.87,
  "average_prediction_accuracy": 0.82,
  "alignment_scores": [0.88, 0.85, 0.91, ...]
}
```

---

## 💡 使用场景

### 场景1: 长时间实验
```bash
# 启动实验（可能运行几小时）
python run_prism_pipeline.py \
    --stage trace \
    --n-users 200 \
    --run-id long_experiment \
    --print \
    --resume

# 如果中途失败或中断，重新运行相同命令即可继续
# 已完成的用户不会重新处理
```

### 场景2: 调试特定用户
```bash
# 1. 运行一小批用户，启用详细日志
python run_prism_pipeline.py \
    --stage trace \
    --n-users 5 \
    --run-id debug_run \
    --print

# 2. 查看某个用户的日志
cat preference_results/logs/debug_run/user_001.log

# 3. 查看该用户的指标
python -c "
import json
with open('preference_results/metrics/debug_run/all_metrics.json') as f:
    data = json.load(f)
    print(json.dumps(data['users']['user_001'], indent=2))
"
```

### 场景3: 批量分析
```bash
# 1. 运行实验
python run_prism_pipeline.py \
    --stage trace \
    --n-users 100 \
    --run-id batch_analysis \
    --resume

# 2. 分析所有指标
python analyze_metrics.py \
    --metrics-file preference_results/metrics/batch_analysis/all_metrics.json \
    --output analysis_report.html
```

### 场景4: A/B测试
```bash
# 实验A: 4个假设
python run_prism_pipeline.py \
    --stage trace \
    --n-users 50 \
    --n-hypotheses 4 \
    --run-id experiment_4hyp \
    --resume

# 实验B: 8个假设
python run_prism_pipeline.py \
    --stage trace \
    --n-users 50 \
    --n-hypotheses 8 \
    --run-id experiment_8hyp \
    --resume

# 比较结果
python compare_experiments.py \
    --exp1 preference_results/metrics/experiment_4hyp/all_metrics.json \
    --exp2 preference_results/metrics/experiment_8hyp/all_metrics.json
```

---

## 🔧 高级功能

### 查看检查点状态

```python
from checkpoint_manager import CheckpointManager

checkpoint_mgr = CheckpointManager('preference_results/checkpoints', 'my_experiment')

# 获取进度
progress = checkpoint_mgr.get_progress_summary()
print(f"Completed: {progress['completed']}")
print(f"Failed: {progress['failed']}")

# 获取已完成用户列表
completed = checkpoint_mgr.get_completed_users()
print(f"Completed users: {completed}")

# 获取失败用户列表
failed = checkpoint_mgr.get_failed_users()
print(f"Failed users: {failed}")
```

### 读取和分析指标

```python
import json
from data_manager import DataManager

# 加载指标
with open('preference_results/metrics/my_experiment/all_metrics.json') as f:
    metrics = json.load(f)

# 分析特定用户
user_data = metrics['users']['user_001']
print(f"Turns: {len(user_data['turns'])}")
print(f"Final alignment: {user_data['final_alignment']['score']:.3f}")

# 计算平均准确率
accuracies = []
for user_data in metrics['users'].values():
    for turn in user_data['turns']:
        if 'prediction_correct' in turn:
            accuracies.append(turn['prediction_correct'])

avg_accuracy = sum(accuracies) / len(accuracies)
print(f"Average prediction accuracy: {avg_accuracy:.3f}")
```

### 手动重置特定用户

```python
from checkpoint_manager import CheckpointManager
import json

checkpoint_mgr = CheckpointManager('preference_results/checkpoints', 'my_experiment')

# 从已完成列表中移除特定用户
checkpoint_mgr.checkpoint['completed_users'].remove('user_042')
checkpoint_mgr.checkpoint['user_status'].pop('user_042', None)
checkpoint_mgr._save_checkpoint()

print("User user_042 reset - will be reprocessed on next run")
```

---

## 📈 监控运行状态

### 实时查看进度

```bash
# 在另一个终端监控检查点
watch -n 5 'python -c "
import json
with open(\"preference_results/checkpoints/checkpoint_my_experiment.json\") as f:
    cp = json.load(f)
print(f\"Completed: {len(cp[\"completed_users\"])}\")
print(f\"Failed: {len(cp[\"failed_users\"])}\")
print(f\"Last updated: {cp[\"last_updated\"]}\")
"'
```

### 查看日志输出

```bash
# 实时查看最新用户的日志
tail -f preference_results/logs/my_experiment/user_*.log
```

---

## ⚠️ 注意事项

1. **检查点是自动的**
   - 每个用户处理开始、完成、失败时自动保存
   - 不需要手动管理

2. **指标数据始终保存**
   - `all_metrics.json` 无论是否使用 `--print` 都会保存
   - 包含所有轮次的完整评估数据

3. **日志文件仅在使用 `--print` 时保存**
   - `--print` 同时启用终端显示和文件保存
   - 如果只想保存不显示，可修改 `data_manager.py`

4. **断点续跑的幂等性**
   - 多次运行相同命令，已完成的用户不会重新处理
   - 结果文件会自动合并，不会覆盖

5. **失败处理**
   - 失败的用户会被记录但不会阻止继续处理
   - 可以查看检查点文件了解失败原因

---

## 🎓 最佳实践

1. **使用有意义的 run_id**
   ```bash
   --run-id experiment_4hyp_50users_$(date +%Y%m%d)
   ```

2. **长时间运行使用 nohup**
   ```bash
   nohup python run_prism_pipeline.py \
       --stage trace \
       --n-users 200 \
       --run-id long_run \
       --resume \
       > output.log 2>&1 &
   ```

3. **定期备份结果**
   ```bash
   tar -czf backup_$(date +%Y%m%d).tar.gz preference_results/
   ```

4. **使用配置文件管理不同实验**
   ```yaml
   # configs.yaml
   experiment_baseline:
     n_users: 100
     n_hypotheses: 4
     run_id: baseline_v1
   
   experiment_ablation:
     n_users: 100
     n_hypotheses: 8
     run_id: ablation_v1
   ```

---

## 🐛 故障排查

### 问题: 检查点显示已完成，但想重新运行

```bash
python run_prism_pipeline.py \
    --stage trace \
    --run-id my_experiment \
    --reset-checkpoint
```

### 问题: 某个用户失败，想单独重试

```python
# 手动重置该用户（见"手动重置特定用户"章节）
# 然后重新运行
```

### 问题: 指标文件太大

指标文件实时保存，包含所有细节。如果太大：
- 考虑减少保存的假设数量（修改 `preference_tracer.py` 中的 `hypotheses[:3]`）
- 或在后处理阶段压缩数据

---

## 📚 相关文件

- `checkpoint_manager.py`: 检查点管理实现
- `data_manager.py`: 数据和日志管理实现
- `preference_tracer.py`: 集成了检查点和数据管理的追踪器
- `run_prism_pipeline.py`: 主运行脚本
- `LOGGING_GUIDE.md`: 日志系统使用指南

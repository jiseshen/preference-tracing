# 🚀 快速开始 - 检查点和数据管理

## ✅ 测试系统

首先运行测试，确保一切正常：

```bash
python3 test_checkpoint_system.py
```

应该看到：
```
============================================================
🎉 所有测试通过！
============================================================
```

---

## 📝 基本使用

### 1. 运行追踪（首次）

```bash
python run_prism_pipeline.py \
    --stage trace \
    --n-users 10 \
    --run-id my_experiment \
    --print \
    --resume
```

**参数说明**:
- `--print`: 在终端显示详细日志 + 保存到文件
- `--resume`: 支持断点续跑
- `--run-id`: 实验标识符

### 2. 如果程序中断，继续运行

重新运行完全相同的命令即可！已完成的用户会被跳过。

```bash
# 完全相同的命令
python run_prism_pipeline.py \
    --stage trace \
    --n-users 10 \
    --run-id my_experiment \
    --print \
    --resume
```

会看到：
```
📌 Found existing checkpoint for run 'my_experiment':
   - Completed: 5 users
   - Failed: 0 users

📊 Processing Status:
   Total users: 10
   Completed: 5
   Remaining: 5
```

### 3. 查看结果

```bash
# 查看汇总统计
cat preference_results/summary_my_experiment.json

# 查看所有指标
cat preference_results/metrics/my_experiment/all_metrics.json

# 查看某个用户的日志
cat preference_results/logs/my_experiment/user_001.log

# 查看某个用户的追踪数据
cat preference_results/traces/my_experiment/trace_user_001.json
```

---

## 📊 保存的数据

### 自动保存（无论是否使用 --print）：

1. **指标数据** `metrics/my_experiment/all_metrics.json`
   - 每轮的生成分数
   - 预测准确率（是否预测正确）
   - ESS、diversity
   - 最终对齐分数

2. **检查点** `checkpoints/checkpoint_my_experiment.json`
   - 哪些用户已完成
   - 哪些用户失败
   - 处理进度

3. **追踪数据** `traces/my_experiment/trace_user_*.json`
   - 每个用户的完整追踪数据

### 仅在使用 --print 时保存：

4. **日志文件** `logs/my_experiment/user_*.log`
   - 每个用户的详细处理日志

---

## 🔄 常用命令

### 重置检查点（从头开始）

```bash
python run_prism_pipeline.py \
    --stage trace \
    --n-users 10 \
    --run-id my_experiment \
    --reset-checkpoint
```

### 不显示日志，只保存数据

```bash
python run_prism_pipeline.py \
    --stage trace \
    --n-users 10 \
    --run-id my_experiment \
    --resume
```

（去掉 `--print`，指标数据依然会保存）

### 使用配置文件

编辑 `configs.yaml`:

```yaml
my_config:
  n_users: 50
  n_hypotheses: 4
  print: true
  run_id: experiment_v1
```

运行：

```bash
python run_prism_pipeline.py \
    --config my_config \
    --stage trace \
    --resume
```

---

## 💡 提示

1. **总是使用 `--resume`**：即使是首次运行也可以用，它会自动判断

2. **数据实时保存**：每完成一个用户就保存一次，中断不会丢失数据

3. **失败不会停止**：某个用户失败不会影响其他用户的处理

4. **查看进度**：
   ```bash
   cat preference_results/checkpoints/checkpoint_my_experiment.json
   ```

5. **完整文档**：`CHECKPOINT_AND_DATA_GUIDE.md` 有详细的使用指南和场景示例

---

## 📚 下一步

- 查看 `CHECKPOINT_AND_DATA_GUIDE.md` 了解详细功能
- 查看 `LOGGING_GUIDE.md` 了解日志系统
- 运行 `python3 demo_logging.py` 查看日志演示

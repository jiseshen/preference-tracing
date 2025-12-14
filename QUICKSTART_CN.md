# PRISM Preference Tracing - Quick Start Guide

## System Overview

**Tracer 调用逻辑总结：**

1. **Initialize (初始化)**: 从第一次交互生成 n 个关于用户偏好的假设
   - 输入：对话历史 + 候选回复
   - 输出：HypothesesSetV3 对象，包含 n 个假设和均匀权重

2. **Propagate (传播)**: 基于新的上下文更新每个假设
   - 对每个现有假设，根据新的对话轮次进行更新
   - 保持父子关系链，用于回溯

3. **Weigh (加权)**: 评估用户选择在每个假设下的可能性
   - 使用 LLM 评估：给定假设 H，用户选择 c 的可能性
   - 6档评分 (Very Likely 到 Very Unlikely)
   - 通过 softmax 归一化得到权重

4. **Resample (重采样)**: 当 ESS < threshold 时重采样
   - ESS = 1 / Σ(w_i²)
   - 阈值：n_hypotheses / 2
   - 防止假设退化

5. **Summarize (总结)**: 将加权假设聚合为用户画像
   - 综合所有假设（考虑权重）
   - 生成连贯的用户偏好档案

## 与原始 Tracer 的关键差异

| 维度 | 原始 Tracer (ToM) | PRISM Preference Tracer |
|------|-------------------|-------------------------|
| **目标** | 追踪 agent 的心理状态 | 追踪用户偏好 |
| **Perception** | agent 的感知（视觉、听觉等） | 对话历史 + 候选回复 |
| **Action** | agent 的行为（移动、说话等） | 用户的选择（选中某个回复） |
| **Hypothesis** | agent 的信念、意图 | 用户偏好档案（价值观、风格） |
| **Context** | 故事场景 | 多轮对话 |

## 快速开始

### 1. 环境设置

```bash
# 安装依赖
pip install -r requirements_prism.txt

# 设置 API key
export OPENAI_API_KEY="your-openai-key"
```

### 2. 验证设置

```bash
python test_setup.py
```

如果所有测试通过，你应该看到：
```
✓ All tests passed! System is ready.
```

### 3. 运行完整流程（小规模测试）

```bash
python run_prism_pipeline.py \
    --stage all \
    --tracing-model gpt-4o-mini \
    --n-hypotheses 4 \
    --n-users 5 \
    --run-id test_run
```

这将：
- 处理 5 个用户的对话
- 每个用户维护 4 个偏好假设
- 生成所有评估和可视化

### 4. 查看结果

结果保存在 `preference_results/` 目录：

```bash
preference_results/
├── preference_tracing_results_test_run.json    # 详细结果
├── preference_tracing_summary_test_run.json    # 统计摘要
├── survey_evaluation_test_run.json             # 问卷对齐评估
└── plots/
    ├── learning_curves.png                     # 学习曲线
    ├── user_trajectories.png                   # 用户轨迹
    └── survey_alignment.png                    # 问卷对齐
```

## 分阶段运行

### 仅运行偏好追踪

```bash
python run_prism_pipeline.py \
    --stage trace \
    --tracing-model gpt-4o-mini \
    --n-hypotheses 4 \
    --n-users 10 \
    --run-id experiment_1
```

### 仅评估问卷对齐

```bash
python run_prism_pipeline.py \
    --stage evaluate \
    --eval-model gpt-4o-mini \
    --run-id experiment_1
```

### 仅生成可视化

```bash
python run_prism_pipeline.py \
    --stage visualize \
    --run-id experiment_1
```

## 核心算法流程

```python
for user in users:
    hypotheses = None
    
    for turn in user_conversations:
        # 获取当前轮次数据
        history = conversations[:turn]
        candidates = turn.model_responses
        chosen = turn.chosen_response
        
        # 第一轮：初始化假设
        if turn == 0:
            hypotheses = initialize_hypotheses(history, candidates)
        # 后续轮次：传播假设
        else:
            hypotheses = propagate_hypotheses(hypotheses, history, candidates)
        
        # 基于用户选择加权假设
        weights = weigh_hypotheses(hypotheses, chosen, candidates)
        hypotheses.update_weights(weights)
        
        # 检查是否需要重采样
        ess = compute_ess(hypotheses)
        if ess < n_hypotheses / 2:
            hypotheses = resample_hypotheses(hypotheses)
        
        # 生成当前用户画像
        user_profile = summarize_hypotheses(hypotheses)
        
        # 评估性能
        gen_score = evaluate_generation(user_profile, chosen)
        prediction = predict_choice(user_profile, candidates)
```

## 输出指标说明

### 1. Generation Score (生成质量分数)
- **范围**: 0-1
- **含义**: 基于用户画像生成的回复与用户实际选择的相似度
- **评估**: 独立 LLM 评估风格、内容、对齐度

### 2. Prediction Accuracy (预测准确率)
- **范围**: 0-1 (二值)
- **含义**: 是否正确预测用户会选择哪个候选回复
- **方法**: 基于用户画像对候选排序

### 3. Survey Alignment (问卷对齐分数)
- **范围**: 1-10 (每个维度)
- **维度**:
  - Communication Style: 沟通风格一致性
  - Value Alignment: 价值观对齐度
  - Preference Consistency: 偏好一致性
  - Overall Accuracy: 总体准确度

## 学习曲线解读

理想的学习曲线应该显示：

1. **Generation Score**: 随着轮次增加而上升
   - 说明用户画像越来越准确
   - 能生成更符合用户偏好的回复

2. **Prediction Accuracy**: 随着轮次增加而上升
   - 说明对用户选择的预测越来越准确
   - 偏好学习有效

3. **置信区间**: 应该逐渐缩小
   - 说明不同用户的学习效果趋于稳定

## 参数调优建议

| 参数 | 较小值 | 较大值 | 影响 |
|------|--------|--------|------|
| `n-hypotheses` | 2-3 | 6-8 | 假设多样性 vs 计算成本 |
| `n-users` | 5-10 (测试) | 50-100 (完整) | 统计显著性 |
| 模型选择 | gpt-4o-mini | gpt-4o | 质量 vs 成本 |

### 推荐配置

**快速测试**:
```bash
--n-hypotheses 3 --n-users 5 --tracing-model gpt-4o-mini
```

**标准实验**:
```bash
--n-hypotheses 4 --n-users 20 --tracing-model gpt-4o-mini
```

**高质量实验**:
```bash
--n-hypotheses 6 --n-users 50 --tracing-model gpt-4o
```

## 常见问题

### Q: 如何减少 API 调用成本？
A: 
- 使用 gpt-4o-mini 而不是 gpt-4o
- 减少 n-hypotheses 数量
- 先用少量用户测试

### Q: 学习曲线没有上升趋势？
A: 可能原因：
- n-hypotheses 太少（增加到 4-6）
- 用户对话轮次太少
- 模型能力不足（尝试更强模型）

### Q: 如何处理多模态用户？
A: 系统已设计为用户级别，每个用户独立追踪

### Q: 可以使用本地模型吗？
A: 可以，修改 `load_model` 以支持本地模型（参考 thought-tracing/agents/）

## 下一步

1. **扩展实验**: 增加用户数量和假设数量
2. **分析结果**: 查看哪些用户类型更容易学习
3. **改进假设生成**: 自定义 `initialize_hypotheses()` 方法
4. **添加新指标**: 扩展评估维度

## 技术支持

- 代码问题：检查 `test_setup.py` 输出
- API 问题：确认 `OPENAI_API_KEY` 设置正确
- 数据问题：确认能访问 HuggingFace datasets

祝实验顺利！🚀

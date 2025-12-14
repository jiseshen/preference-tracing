# PRISM Preference Tracing - Implementation Summary

## 核心思想

将 thought-tracing 的 Theory of Mind (ToM) 追踪框架应用于用户偏好学习，通过粒子滤波维护多个偏好假设，在线更新用户画像。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    PRISM Preference Tracer                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  For each user:                                             │
│    ┌─────────────────────────────────────────────┐         │
│    │ For each conversation turn:                 │         │
│    │   1. Observe: history + candidates          │         │
│    │   2. Initialize/Propagate hypotheses        │         │
│    │   3. Weigh by user's choice likelihood      │         │
│    │   4. Resample if ESS < threshold            │         │
│    │   5. Summarize → user profile               │         │
│    │   6. Evaluate: generation + prediction      │         │
│    └─────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Aggregation & Analysis                           │
│  • Average scores across users per turn                     │
│  • Compute confidence intervals                             │
│  • Generate learning curves                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Survey Evaluation (Optional)                     │
│  • Compare inferred profiles with survey data               │
│  • Evaluate: style, values, consistency                     │
└─────────────────────────────────────────────────────────────┘
```

## 文件结构

```
preference-tracing/
├── preference_tracer.py          # 核心追踪引擎
├── survey_evaluator.py           # 问卷对齐评估
├── visualize_results.py          # 结果可视化
├── run_prism_pipeline.py         # 主运行脚本
├── test_setup.py                 # 设置验证脚本
├── requirements_prism.txt        # Python 依赖
├── PRISM_README.md              # 完整文档
├── QUICKSTART_CN.md             # 快速开始指南（中文）
└── thought-tracing/             # 原始 tracer 框架
    ├── tracer.py
    ├── hypothesis.py
    ├── utils.py
    └── agents/
```

## PreferenceTracer 核心方法

### 1. initialize_hypotheses()
```python
输入: user_id, conversation_history, candidates
处理:
  - 格式化对话历史和候选回复
  - 提示 LLM 生成 n 个偏好假设
  - 初始化均匀权重
输出: HypothesesSetV3 对象
```

### 2. propagate_hypotheses()
```python
输入: existing_hypotheses, new_conversation_history, new_candidates
处理:
  - 对每个现有假设，提示 LLM 基于新上下文更新
  - 维护假设的父子链接（用于回溯）
输出: 更新后的 HypothesesSetV3
```

### 3. weigh_hypotheses()
```python
输入: hypotheses, chosen_response, candidates
处理:
  - 对每个假设 h:
    - 提示: "给定用户偏好 h，选择 chosen_response 的可能性？"
    - LLM 回答: 6档量表 (Very Likely ~ Very Unlikely)
  - 映射到数值分数
  - Softmax 归一化得到权重
输出: {weights, prompts, reasonings, raw_scores}
```

### 4. summarize_hypotheses()
```python
输入: weighted hypotheses
处理:
  - 列出所有假设及其权重
  - 提示 LLM 综合为连贯的用户画像
  - 强调高权重假设
输出: 用户偏好档案字符串
```

### 5. evaluate_generation()
```python
输入: user_profile, conversation_history, chosen_response
处理:
  - 基于 user_profile 生成回复
  - 用 eval_model 比较生成回复和实际选择
  - 评分 1-10，归一化到 0-1
输出: generation_score (float)
```

### 6. predict_choice()
```python
输入: user_profile, conversation_history, candidates
处理:
  - 基于 user_profile 预测最可能选择
  - 返回候选索引
输出: predicted_index (int)
```

### 7. trace_user_preferences()
```python
输入: user_conversations (list of conversation dicts)
处理:
  - 按轮次分组对话
  - 逐轮应用 initialize/propagate/weigh/resample/summarize
  - 每轮评估 generation 和 prediction
输出: {user_id, turn_results, final_profile}
```

## 数据流

### PRISM 数据集结构
```json
{
  "conversation_id": "c1234",
  "user_id": "u5678",
  "turn": 0,
  "role": "user",
  "content": "im redoing a task i completed yesterday! is that fair?",
  "model_provider": null,
  "model_name": null,
  "score": null,
  "if_chosen": null,
  "within_turn_id": null
}
```

多条记录形成一个轮次（turn）：
- 1 条 user 消息
- 多条 model 回复（不同模型）
- 其中一条 if_chosen=True

### 内部表示

**HypothesesSetV3**:
```python
{
  'target_agent': 'user_id',
  'contexts': [context_dict_per_turn],
  'perceptions': [perception_dict_per_turn],
  'texts': ['hypothesis 1', 'hypothesis 2', ...],
  'weights': [w1, w2, ...],  # sum to 1
  'hypotheses': [HypothesisV3 objects]  # 包含父链接
}
```

**Turn Results**:
```python
{
  'turn': 0,
  'user_profile': 'summarized profile',
  'gen_score': 0.85,
  'prediction_correct': True,
  'hypotheses': ['h1', 'h2', 'h3', 'h4'],
  'weights': [0.4, 0.3, 0.2, 0.1]
}
```

## 关键算法细节

### Effective Sample Size (ESS)
```python
ESS = 1 / Σ(w_i²)

# 退化检测
if ESS < n_hypotheses / 2:
    hypotheses = resample_hypotheses_with_other_info(hypotheses, ESS)
```

- ESS 接近 n 时：权重均匀，假设多样性好
- ESS 接近 1 时：单个假设主导，假设退化
- 重采样：根据权重采样，重置为均匀权重

### 权重计算

```python
# LLM 评分映射
score_mapping = {
    'a': 3,      # Very Likely (90%)
    'b': 2.5,    # Likely (70%)
    'c': 2,      # Somewhat Likely (50%)
    'd': 1,      # Somewhat Unlikely (30%)
    'e': 0.5,    # Unlikely (10%)
    'f': 0.001   # Very Unlikely (<5%)
}

# Softmax 归一化
raw_scores = [map_response(r, score_mapping) for r in llm_responses]
weights = softmax(raw_scores)
```

### Hypothesis 链路追踪

每个 HypothesisV3 对象维护：
- `parent`: 指向上一轮的父假设
- `text`: 当前假设文本
- `weight`: 当前权重
- `contexts`: 上下文列表
- `perceptions`: 感知列表

可通过 `backtrack(hypothesis)` 回溯整个演化路径。

## 评估指标实现

### 1. Generation Score
```python
# 1. 基于用户画像生成回复
generated = tracer_model.interact(
    f"User profile: {profile}\nHistory: {history}\nGenerate response:"
)

# 2. 评估与实际选择的相似度
rating = eval_model.interact(
    f"Generated: {generated}\nActual: {chosen}\nRate similarity 1-10:"
)

score = rating / 10.0  # 归一化到 0-1
```

### 2. Prediction Accuracy
```python
# 1. 基于用户画像预测选择
predicted_idx = tracer_model.interact(
    f"User profile: {profile}\nCandidates: {candidates}\nWhich would user choose? (1-n):"
)

# 2. 与实际选择比较
actual_idx = find_chosen_index(candidates)
accuracy = 1.0 if predicted_idx == actual_idx else 0.0
```

### 3. Survey Alignment
```python
# 1. 提取问卷信息
survey_profile = extract_survey_profile(survey_data)

# 2. 多维度评估
for dimension in ['communication_style', 'value_alignment', ...]:
    score = eval_model.rate_alignment(inferred_profile, survey_profile, dimension)
```

## 可扩展性设计

### 1. 自定义假设生成器
```python
class CustomPreferenceTracer(PreferenceTracer):
    def initialize_hypotheses(self, user_id, history, candidates):
        # 自定义假设生成逻辑
        # 例如：基于用户人口统计信息、历史行为等
        pass
```

### 2. 替代加权方案
```python
def weigh_hypotheses_custom(self, hypotheses, chosen, candidates):
    # 例如：基于嵌入相似度、强化学习等
    pass
```

### 3. 新评估指标
```python
def evaluate_diversity(self, user_profile, candidates):
    # 评估生成的多样性
    pass

def evaluate_safety(self, user_profile, generated):
    # 评估安全性对齐
    pass
```

## 性能优化

### Batch Processing
```python
# propagate_hypotheses 内部使用 batch_interact
propagated_texts = self.tracer_model.batch_interact(
    propagation_prompts,  # 所有假设的 prompts
    temperature=0,
    max_tokens=512
)
```

### Caching
可添加缓存层：
```python
@lru_cache(maxsize=1000)
def cached_interact(prompt_hash):
    return model.interact(prompt)
```

## 实验建议

### 消融实验
1. **假设数量**: n_hypotheses = [2, 4, 6, 8]
2. **重采样阈值**: threshold = [0.25, 0.5, 0.75] * n_hypotheses
3. **模型选择**: gpt-4o-mini vs gpt-4o
4. **提示工程**: 不同的假设生成提示

### 分析维度
1. **用户类型**: 按问卷数据分组分析
2. **对话长度**: 短 vs 长对话的学习效果
3. **响应多样性**: 候选回复差异对学习的影响
4. **时间动态**: 用户偏好是否随时间变化

## 限制与改进方向

### 当前限制
1. **计算成本**: n_hypotheses * n_turns * LLM calls
2. **提示依赖**: 性能依赖提示质量
3. **静态假设**: 假设空间固定，不自适应

### 改进方向
1. **自适应采样**: 动态调整 n_hypotheses
2. **主动学习**: 选择最具信息量的轮次
3. **层次化假设**: 多层次的偏好表示
4. **多模态**: 整合对话之外的信号

## 总结

PRISM Preference Tracer 成功将 ToM 追踪框架应用于偏好学习：

✓ **在线学习**: 逐轮更新用户画像
✓ **不确定性量化**: 多假设 + 权重
✓ **可解释性**: 可追踪假设演化
✓ **评估完善**: 多维度评估指标
✓ **可扩展**: 模块化设计，易于定制

核心创新在于将"追踪 agent 心理状态"转换为"追踪用户偏好"，利用用户的选择行为作为观察信号，通过粒子滤波实现在线偏好学习。

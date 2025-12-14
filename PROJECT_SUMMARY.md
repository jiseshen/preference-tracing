# PRISM Preference Tracing - Complete System Documentation

## 项目概述

本项目将 thought-tracing 框架（用于 Theory of Mind 推理）改造为用户偏好在线学习系统，应用于 PRISM pluralistic alignment 数据集。

### 核心创新
- **从 ToM 到偏好学习**: 将"追踪 agent 心理状态"转换为"追踪用户偏好"
- **在线学习**: 逐轮对话更新用户画像，无需批量训练
- **不确定性建模**: 维护多个假设及其权重，量化不确定性
- **可解释性**: 每个决策都可追溯到具体的偏好假设

---

## 文件清单

### 核心代码
| 文件 | 功能 | 行数 |
|------|------|------|
| `preference_tracer.py` | 主要追踪引擎 | ~350 |
| `survey_evaluator.py` | 问卷对齐评估 | ~150 |
| `visualize_results.py` | 结果可视化 | ~150 |
| `run_prism_pipeline.py` | 主运行脚本 | ~100 |
| `batch_runner.py` | 批量实验运行器 | ~200 |

### 文档
| 文件 | 内容 |
|------|------|
| `PRISM_README.md` | 完整英文文档 |
| `QUICKSTART_CN.md` | 快速开始指南（中文） |
| `IMPLEMENTATION_SUMMARY.md` | 实现细节总结 |
| `PROJECT_SUMMARY.md` | 本文档 |

### 配置与测试
| 文件 | 功能 |
|------|------|
| `configs.yaml` | 实验配置模板 |
| `requirements_prism.txt` | Python 依赖 |
| `test_setup.py` | 环境验证脚本 |

---

## Tracer 调用逻辑详解

### 原始 Tracer (thought-tracing/tracer.py)

```python
# 1. 预处理输入 - 识别目标 agent 和标注动作
preprocessed = preprocess_input(text, target_agent)
trajectory = preprocessed['trajectory']  # state-action 序列
perceptions = preprocessed['perceptions']  # 每步的感知

# 2. 逐步追踪
hypotheses_list = []
for state_action, perception in zip(trajectory, perceptions):
    if first_step:
        # 初始化: 生成 n 个关于 agent 信念的假设
        hypotheses = initialize(state_action, perception)
    else:
        # 传播: 基于新观察更新每个假设
        hypotheses = propagate(existing_hypotheses, state_action, perception)
    
    if state_action['action']:
        # 加权: 基于动作可能性评估每个假设
        weights = weigh(hypotheses, action, mode="prompting")
        hypotheses.update_weights(weights)
        
        # 重采样: 如果假设退化（ESS 太低）
        ess = compute_ess(hypotheses)
        if ess < threshold:
            hypotheses = resample_hypotheses(hypotheses, ess)
    
    hypotheses_list.append(hypotheses)

# 3. 汇总输出
traced_thoughts = chain_weighted_average_trace(hypotheses_list)
```

**关键方法**:

1. **initialize()**: 
   - 输入: state, action, perception
   - 提示: "生成 n 个假设，解释 agent 的行为"
   - 输出: n 个均匀权重的假设

2. **propagate()**:
   - 输入: 现有假设 + 新观察
   - 提示: "给定之前的假设和新信息，agent 现在相信什么？"
   - 输出: 每个假设的更新版本

3. **weigh()**:
   - 输入: 假设 + 实际观察到的动作
   - 提示: "给定假设 H，agent 执行动作 A 的可能性？"
   - LLM 回答 6 档量表，映射到分数，softmax 归一化

4. **resample()**:
   - 计算 ESS = 1 / Σ(w_i²)
   - 如果 ESS < n/2，按权重重采样 n 个假设
   - 重置权重为均匀分布

5. **chain_weighted_average_trace()**:
   - 对每一步，用权重平均所有假设
   - 生成连贯的思维链叙述

---

## PRISM Preference Tracer 适配

### 关键映射

| 维度 | ToM Tracer | Preference Tracer |
|------|-----------|-------------------|
| **目标** | Agent (故事中的角色) | User (对话参与者) |
| **状态** | 场景描述 | 对话历史 |
| **动作** | Agent 行为 | User 选择某个回复 |
| **感知** | 视觉、听觉等感官信息 | 可见的回复候选 |
| **假设** | Agent 的信念、意图 | User 的偏好、价值观 |

### 核心方法适配

#### 1. initialize_hypotheses()
```python
def initialize_hypotheses(self, user_id, conversation_history, candidates):
    # 格式化输入
    history_str = format_conversation_history(conversation_history)
    candidates_str = format_candidates(candidates)
    
    # 构造提示
    prompt = f"""
    <conversation history>
    {history_str}
    </conversation history>
    
    <current responses>
    {candidates_str}
    </current responses>
    
    Generate {n_hypotheses} hypotheses about the user's preferences, 
    values, and communication style that would explain their response choice.
    """
    
    # 生成假设
    hypotheses_list = prompting_for_ordered_list(tracer_model, prompt, n_hypotheses)
    weights = uniform(n_hypotheses)
    
    return HypothesesSetV3(user_id, contexts, perceptions, hypotheses_list, weights)
```

**示例输出**:
```
1. User prefers concise, direct responses without unnecessary elaboration
2. User values factual accuracy over politeness in technical discussions
3. User appreciates when AI acknowledges uncertainty rather than overconfident responses
4. User prefers responses that provide actionable next steps
```

#### 2. propagate_hypotheses()
```python
def propagate_hypotheses(self, existing_hypotheses, new_history, new_candidates):
    # 对每个现有假设
    propagation_prompts = []
    for hypothesis in existing_hypotheses.texts:
        prompt = f"""
        <previous user preference hypothesis>
        {hypothesis}
        </previous user preference hypothesis>
        
        <new conversation context>
        {format_new_context(new_history, new_candidates)}
        </new conversation context>
        
        Question: Based on the previous hypothesis and new conversation context,
        what are the user's updated preferences and values? Provide a refined hypothesis.
        """
        propagation_prompts.append(prompt)
    
    # 批量处理
    propagated_texts = tracer_model.batch_interact(propagation_prompts)
    
    return HypothesesSetV3(..., texts=propagated_texts, weights=existing_weights, 
                          parent_hypotheses=existing_hypotheses.hypotheses)
```

#### 3. weigh_hypotheses()
```python
def weigh_hypotheses(self, hypotheses, chosen_response, candidates):
    # 构造评估提示
    likelihood_prompts = []
    for hypothesis in hypotheses.texts:
        prompt = f"""
        <user preference hypothesis>
        {hypothesis}
        </user preference hypothesis>
        
        <available responses>
        {format_candidates(candidates, include_chosen=True)}
        </available responses>
        
        <chosen response>
        {chosen_response['content']}
        </chosen response>
        
        Question: Given the user's preferences described in the hypothesis,
        how likely would they choose the response above?
        
        Options:
        (a) Very Likely (90%)
        (b) Likely (70%)
        (c) Somewhat Likely (50%)
        (d) Somewhat Unlikely (30%)
        (e) Unlikely (10%)
        (f) Very Unlikely (<5%)
        
        Briefly explain and then answer with one option.
        Answer:
        """
        likelihood_prompts.append(prompt)
    
    # 批量评估
    raw_predictions = tracer_model.batch_interact(likelihood_prompts)
    
    # 解析和映射分数
    score_mapping = {'a': 3, 'b': 2.5, 'c': 2, 'd': 1, 'e': 0.5, 'f': 0.001}
    raw_scores = [map_response_to_score(pred, score_mapping) for pred in raw_predictions]
    weights = softmax(raw_scores)
    
    return {'weights': weights, 'prompts': likelihood_prompts, 
            'raw_predictions': raw_predictions, 'raw_scores': raw_scores}
```

#### 4. summarize_hypotheses()
```python
def summarize_hypotheses(self, hypotheses):
    # 列出加权假设
    weighted_list = "\n".join([
        f"- {text} (weight: {weight:.3f})"
        for text, weight in zip(hypotheses.texts, hypotheses.weights)
    ])
    
    prompt = f"""
    <weighted hypotheses about user preferences>
    {weighted_list}
    </weighted hypotheses about user preferences>
    
    Summarize these hypotheses into a coherent user preference profile.
    Focus on the most important preferences (higher weights) and synthesize
    overlapping themes. Provide a concise profile.
    """
    
    summary = tracer_model.interact(prompt, temperature=0, max_tokens=512)
    return summary
```

**示例输出**:
```
This user demonstrates a clear preference for directness and efficiency in 
communication, valuing concise responses that get to the point quickly. They 
prioritize factual accuracy and appreciate when the AI acknowledges limitations
or uncertainty rather than providing overconfident answers. The user tends to
favor actionable advice and practical next steps over abstract explanations.
Their interaction style suggests they value their time and prefer responses
that respect this by being both informative and succinct.
```

#### 5. evaluate_generation()
```python
def evaluate_generation(self, user_profile, conversation_history, 
                       candidates, chosen_response):
    # 1. 基于用户画像生成回复
    gen_prompt = f"""
    <user preference profile>
    {user_profile}
    </user preference profile>
    
    <conversation history>
    {format_conversation_history(conversation_history)}
    </conversation history>
    
    Based on the user's preferences, generate an appropriate response
    to their last message.
    """
    
    generated_response = tracer_model.interact(gen_prompt, temperature=0.7)
    
    # 2. 评估与实际选择的相似度
    eval_prompt = f"""
    <generated response>
    {generated_response}
    </generated response>
    
    <actual chosen response>
    {chosen_response['content']}
    </actual chosen response>
    
    Rate how similar these responses are in style, content, and alignment
    with user preferences on a scale of 1-10.
    
    Rating:
    """
    
    rating_response = eval_model.interact(eval_prompt, temperature=0)
    rating = parse_rating(rating_response)  # 提取数值
    
    return rating / 10.0  # 归一化到 0-1
```

#### 6. predict_choice()
```python
def predict_choice(self, user_profile, conversation_history, candidates):
    candidates_str = format_candidates(candidates, numbered=True)
    
    prompt = f"""
    <user preference profile>
    {user_profile}
    </user preference profile>
    
    <conversation history>
    {format_conversation_history(conversation_history)}
    </conversation history>
    
    <response candidates>
    {candidates_str}
    </response candidates>
    
    Based on the user's preferences, which response would they most likely
    choose? Answer with just the number (1-{len(candidates)}).
    
    Answer:
    """
    
    prediction = tracer_model.interact(prompt, temperature=0, max_tokens=10)
    predicted_idx = parse_index(prediction) - 1  # 转换为 0-based index
    
    return predicted_idx
```

---

## 完整追踪流程

```python
def trace_user_preferences(self, user_conversations):
    user_id = user_conversations[0]['user_id']
    conversations = user_conversations[0]['conversation']
    
    # 按 turn 分组
    turns = group_by_turns(conversations)
    
    hypotheses = None
    turn_results = []
    
    for turn_idx, turn_data in enumerate(turns):
        # 提取当前轮次数据
        user_msg = turn_data['user_message']
        candidates = turn_data['candidates']  # 多个模型回复
        chosen_response = turn_data['chosen']  # 用户选择的回复
        
        # 构建历史（只包括 chosen 回复）
        history = build_history(turns[:turn_idx + 1])
        
        # 步骤 1: 初始化或传播假设
        if hypotheses is None:
            hypotheses = initialize_hypotheses(user_id, history, candidates)
        else:
            hypotheses = propagate_hypotheses(hypotheses, history, candidates)
        
        # 步骤 2: 加权假设
        weight_results = weigh_hypotheses(hypotheses, chosen_response, candidates)
        hypotheses.update_weights(weight_results['weights'])
        hypotheses.weight_details = weight_results
        
        # 步骤 3: 检查是否需要重采样
        if n_hypotheses > 1:
            ess = compute_ess(hypotheses)
            if ess < n_hypotheses / 2:
                hypotheses = resample_hypotheses_with_other_info(hypotheses, ess)
        
        # 步骤 4: 生成当前用户画像
        user_profile = summarize_hypotheses(hypotheses)
        
        # 步骤 5: 评估性能
        gen_score = evaluate_generation(user_profile, history, 
                                       candidates, chosen_response)
        predicted_idx = predict_choice(user_profile, history, candidates)
        actual_idx = find_chosen_index(candidates)
        prediction_correct = (predicted_idx == actual_idx)
        
        # 保存结果
        turn_results.append({
            'turn': turn_idx,
            'user_profile': user_profile,
            'gen_score': gen_score,
            'prediction_correct': prediction_correct,
            'hypotheses': hypotheses.texts,
            'weights': hypotheses.weights.tolist()
        })
    
    return {
        'user_id': user_id,
        'turn_results': turn_results,
        'final_profile': turn_results[-1]['user_profile'] if turn_results else ""
    }
```

---

## 使用指南

### 1. 快速开始

```bash
# 验证环境
python test_setup.py

# 小规模测试（5 用户）
python run_prism_pipeline.py --stage all --n-users 5 --run-id test

# 查看结果
ls preference_results/
# preference_tracing_results_test.json
# preference_tracing_summary_test.json
# survey_evaluation_test.json
# plots/
```

### 2. 使用配置文件

```bash
# 单个实验
python batch_runner.py --experiments quick_test

# 消融实验
python batch_runner.py --ablation-studies ablation_hypotheses

# 运行所有
python batch_runner.py
```

### 3. 自定义实验

编辑 `configs.yaml`:
```yaml
my_experiment:
  tracing_model: "gpt-4o-mini"
  eval_model: "gpt-4o-mini"
  n_hypotheses: 5
  n_users: 15
  output_dir: "results/my_exp"
  run_id: "my_exp_v1"
```

运行:
```bash
python batch_runner.py --experiments my_experiment
```

---

## 输出说明

### preference_tracing_results_{run_id}.json

```json
[
  {
    "user_id": "u123",
    "turn_results": [
      {
        "turn": 0,
        "user_profile": "用户画像文本",
        "gen_score": 0.85,
        "prediction_correct": true,
        "hypotheses": ["假设1", "假设2", "假设3", "假设4"],
        "weights": [0.4, 0.3, 0.2, 0.1]
      },
      {
        "turn": 1,
        ...
      }
    ],
    "final_profile": "最终用户画像"
  },
  ...
]
```

### preference_tracing_summary_{run_id}.json

```json
{
  "turn_gen_scores": {
    "0": {"mean": 0.72, "std": 0.15, "ci": 0.08},
    "1": {"mean": 0.78, "std": 0.12, "ci": 0.06},
    ...
  },
  "turn_pred_accuracy": {
    "0": {"mean": 0.45, "std": 0.28, "ci": 0.14},
    "1": {"mean": 0.62, "std": 0.24, "ci": 0.12},
    ...
  }
}
```

### 可视化

- `learning_curves.png`: 分数随轮次变化（带置信区间）
- `user_trajectories.png`: 个体用户的学习曲线
- `survey_alignment.png`: 与问卷数据的对齐程度

---

## 性能与成本

### API 调用估算

每个用户，每轮对话:
- Initialize: n_hypotheses 次调用
- Propagate: n_hypotheses 次调用
- Weigh: n_hypotheses 次调用
- Summarize: 1 次调用
- Evaluate: 2 次调用

**总计**: 约 (3*n_hypotheses + 3) 次/轮

示例 (n_hypotheses=4, avg_turns=3):
- 每用户: ~45 次调用
- 20 用户: ~900 次调用
- gpt-4o-mini: ~$0.90
- gpt-4o: ~$9.00

### 运行时间

- gpt-4o-mini: ~2-3 分钟/用户
- gpt-4o: ~3-5 分钟/用户
- 20 用户: ~40-100 分钟

---

## 扩展与改进

### 1. 自定义假设生成

```python
class CustomPreferenceTracer(PreferenceTracer):
    def initialize_hypotheses(self, user_id, history, candidates):
        # 整合用户人口统计信息
        user_demo = get_user_demographics(user_id)
        
        prompt = f"""
        User demographics: {user_demo}
        Conversation: {history}
        Candidates: {candidates}
        
        Generate hypotheses about user preferences considering their background.
        """
        # ...
```

### 2. 主动学习

```python
def select_informative_turns(self, all_turns):
    # 选择最具信息量的轮次进行学习
    # 例如：候选回复差异最大的轮次
    scores = [compute_candidate_diversity(turn) for turn in all_turns]
    return select_top_k(all_turns, scores, k=5)
```

### 3. 层次化假设

```python
# 顶层：价值观
high_level_hypotheses = ["values accuracy", "values conciseness", ...]

# 底层：具体偏好
for high_level in high_level_hypotheses:
    low_level_hypotheses = generate_specific_preferences(high_level)
```

---

## 总结

✅ **已实现**:
- 完整的偏好追踪流程
- 多假设粒子滤波
- 在线学习与评估
- 可视化与分析
- 批量实验支持

✅ **优势**:
- 无需标注的偏好数据
- 可解释的偏好演化
- 不确定性量化
- 模块化易扩展

📊 **应用**:
- 个性化对话系统
- 用户画像构建
- 偏好对齐研究
- A/B 测试优化

🚀 **下一步**:
1. 运行消融实验
2. 分析不同用户类型
3. 优化提示工程
4. 探索主动学习策略

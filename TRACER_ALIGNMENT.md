# Preference Tracer 与 Original Tracer 对齐说明

## 完整对齐的功能

### 1. **Rejuvenate 机制** ✅

**原始 Tracer 逻辑：**
```python
if ess < n_hypotheses / 2:
    # ESS 太低 → 重采样
    hypotheses = resample_hypotheses(hypotheses, ess)
elif overall_text_diversity < 0.25:
    # 文本多样性太低 → Rejuvenate
    hypotheses = rejuvenate_hypotheses(hypotheses)
```

**Rejuvenate 做什么：**
- 通过 **paraphrase（改写）** 来增加假设的表达多样性
- 保持语义不变，只改变措辞
- 使用 temperature=1 增加随机性
- 防止假设"塌缩"到几乎相同的表述

**Preference Tracer 实现：**
```python
def rejuvenate_hypotheses(self, existing_hypotheses: HypothesesSetV3):
    system_prompt = "Your task is to paraphrase the following user preference hypothesis. 
                     Keep the meaning intact while rephrasing it."
    revision_prompts = [hypothesis for hypothesis in existing_hypotheses.texts]
    revised_texts = self.tracer_model.batch_interact(
        revision_prompts, 
        system_prompts=system_prompt, 
        temperature=1,  # 高 temperature 增加多样性
        max_tokens=512
    )
    existing_hypotheses.update_texts(revised_texts)
    return existing_hypotheses
```

### 2. **完整的追踪流程** ✅

**原始 Tracer (`_trace` 方法)：**
```python
for state_action, perceptions in zip(trajectory, perceptions_trajectory):
    if idx == 0:
        hypotheses = initialize(state_action, perceptions)
    else:
        hypotheses = propagate(existing_hypotheses, state_action, perceptions)
    
    if state_action['action']:
        # 加权
        weight_results = weigh(hypotheses, action, mode="prompting")
        hypotheses.update_weights(weight_results['weights'])
        
        # 重采样或 Rejuvenate
        if n_hypotheses > 1:
            ess = compute_ess(hypotheses)
            diversity = 1 - overall_jaccard_similarity(hypotheses.texts)
            
            if ess < n_hypotheses / 2:
                hypotheses = resample_hypotheses(hypotheses, ess)
            elif diversity < 0.25:
                hypotheses = rejuvenate_hypotheses(hypotheses)
    
    hypotheses_list.append(hypotheses)
```

**Preference Tracer (`trace_user_preferences` 方法)：**
```python
for turn_idx, turn_data in enumerate(turns):
    # 获取 perception (对话历史 + 候选)
    history = build_history(turns[:turn_idx + 1])
    candidates = turn_data['candidates']
    chosen = turn_data['chosen']
    
    # Initialize 或 Propagate
    if hypotheses is None:
        hypotheses = initialize_hypotheses(user_id, history, candidates)
    else:
        hypotheses = propagate_hypotheses(hypotheses, history, candidates)
    
    # 加权 (基于用户选择的可能性)
    weight_results = weigh_hypotheses(hypotheses, chosen, candidates)
    hypotheses.update_weights(weight_results['weights'])
    
    # 重采样或 Rejuvenate (完全对齐！)
    if n_hypotheses > 1:
        ess = compute_ess(hypotheses)
        diversity = 1 - overall_jaccard_similarity(hypotheses.texts)
        
        if ess < n_hypotheses / 2:
            hypotheses = resample_hypotheses_with_other_info(hypotheses, ess)
        elif diversity < 0.25:
            hypotheses = rejuvenate_hypotheses(hypotheses)
    
    hypotheses_list.append(hypotheses)
```

### 3. **Dump 功能** ✅

**原始 Tracer：**
```python
def dump(self, traced_thought: dict, hypotheses_list: List[HypothesesSetV3]):
    dumped_hypotheses_list = [h.dump() for h in hypotheses_list]
    traced_thought['hypotheses'] = dumped_hypotheses_list
    with open(self.output_file, 'a') as f:
        json.dump(traced_thought, f, cls=NpEncoder)
        f.write('\n')
```

**Preference Tracer：**
```python
def dump(self, user_result: Dict, hypotheses_list: List[HypothesesSetV3]):
    dumped_hypotheses_list = [h.dump() for h in hypotheses_list]
    user_result['detailed_hypotheses'] = dumped_hypotheses_list
    
    # 可选：保存详细追踪
    if hasattr(self.args, 'save_detailed_traces') and self.args.save_detailed_traces:
        output_file = os.path.join(
            self.args.output_dir, 
            f"detailed_trace_{user_result['user_id']}_{self.args.run_id}.json"
        )
        with open(output_file, 'w') as f:
            json.dump(user_result, f, indent=2)
    
    return user_result
```

### 4. **Rich 输出和调试信息** ✅

**原始 Tracer：**
```python
from rich import print, box
from rich.panel import Panel

if ess < threshold:
    print(Panel("Resampling...", style="yellow"))
elif diversity < 0.25:
    print(Panel(f"Text diversity: {diversity}", title="Low Variance", style="red"))
    print(Panel("\n".join(hypotheses.texts), title="Hypotheses", style="blue"))
```

**Preference Tracer：**
```python
from rich import print, box
from rich.panel import Panel

# 用户级别追踪开始
if self.args.print:
    print(Panel(f">>> Tracing preferences for user {user_id}", 
                style="blue", box=box.DOUBLE))

# 重采样时
if ess < n_hypotheses / 2:
    if self.args.print:
        print(Panel(f"ESS: {ess:.2f}", title="Resampling Hypotheses", 
                    style="yellow"))

# Rejuvenate 时
elif diversity < 0.25:
    if self.args.print:
        print(Panel(f"Text diversity: {diversity:.3f}", 
                    title="Low Variance Hypotheses", style="red"))
```

### 5. **增强的结果记录** ✅

**Preference Tracer 额外记录：**
```python
turn_results.append({
    'turn': turn_idx,
    'user_profile': user_profile,
    'gen_score': gen_score,
    'prediction_correct': prediction_correct,
    'predicted_idx': predicted_idx,
    'actual_idx': actual_idx,
    'hypotheses': hypotheses.texts,
    'weights': hypotheses.weights.tolist(),
    'ess': compute_ess(hypotheses),              # 新增
    'text_diversity': 1 - overall_jaccard_similarity(hypotheses.texts)  # 新增
})
```

### 6. **数据集加载修复** ✅

**正确加载 PRISM 数据集：**
```python
# 指定 config 名称
dataset = load_dataset("HannahRoseKirk/prism-alignment", "conversations")

# 按 (user_id, conversation_id) 分组
user_conversations = {}
for item in train_data:
    user_id = item['user_id']
    conversation_id = item.get('conversation_id', 'unknown')
    key = (user_id, conversation_id)
    if key not in user_conversations:
        user_conversations[key] = {
            'user_id': user_id,
            'conversation_id': conversation_id,
            'conversation': []
        }
    user_conversations[key]['conversation'].append(item)

# 每个对话按 turn 排序
for conv_data in user_conversations.values():
    conv_data['conversation'].sort(key=lambda x: x.get('turn', 0))
```

## 关键差异（设计上的不同）

### 1. **Perception vs Context**

| 维度 | Original Tracer | Preference Tracer |
|------|----------------|-------------------|
| **Perception 来源** | 视觉、听觉等感官输入 | 对话历史 + 候选回复 |
| **需要 parsing** | ✅ 需要 label_action | ❌ 直接从数据集获取 |
| **State-Action 分离** | ✅ 明确的 state/action | ❌ 统一为对话轮次 |

### 2. **目标追踪对象**

| 维度 | Original Tracer | Preference Tracer |
|------|----------------|-------------------|
| **追踪目标** | Agent 的心理状态 | User 的偏好 |
| **Hypothesis 内容** | "Agent 相信 X" | "User 偏好 Y" |
| **Action** | Agent 的行为 | User 的选择 |

### 3. **评估方式**

| 维度 | Original Tracer | Preference Tracer |
|------|----------------|-------------------|
| **主要评估** | 回答 ToM 问题 | Generation + Prediction |
| **Generation** | ❌ 无 | ✅ 基于画像生成回复 |
| **Prediction** | ❌ 无 | ✅ 预测用户选择 |
| **Survey 对齐** | ❌ 无 | ✅ 与问卷数据对比 |

## 使用示例

### 基础运行
```bash
python preference_tracer.py \
    --tracing-model gpt-4o-mini \
    --eval-model gpt-4o-mini \
    --n-hypotheses 4 \
    --n-users 10 \
    --output-dir results \
    --run-id test
```

### 启用调试输出
```bash
python preference_tracer.py \
    --tracing-model gpt-4o-mini \
    --n-hypotheses 6 \
    --n-users 5 \
    --print \
    --run-id debug_run
```

### 保存详细追踪
```bash
python preference_tracer.py \
    --tracing-model gpt-4o-mini \
    --n-hypotheses 4 \
    --n-users 10 \
    --save-detailed-traces \
    --run-id detailed_run
```

## 完整对齐清单

- [x] **Rejuvenate 机制** - 文本多样性低时改写假设
- [x] **Resample 机制** - ESS 低时重采样
- [x] **Initialize → Propagate → Weigh 流程**
- [x] **Dump 功能** - 保存详细假设演化
- [x] **Rich 输出** - 彩色调试信息
- [x] **ESS 计算和监控**
- [x] **Text diversity 计算和监控**
- [x] **Hypotheses 链式追踪**
- [x] **batch_interact 批处理**
- [x] **数据集正确加载**（conversations config）
- [x] **错误处理** - 用户处理失败时继续

## 总结

Preference Tracer 现在完全对齐了 Original Tracer 的核心机制：

1. ✅ **粒子滤波框架** - Initialize, Propagate, Weigh, Resample
2. ✅ **假设多样性维护** - Resample (ESS) + Rejuvenate (diversity)
3. ✅ **详细追踪记录** - Dump with full hypothesis evolution
4. ✅ **调试可视化** - Rich panels and colored output
5. ✅ **批处理优化** - batch_interact for efficiency

唯一的区别是应用领域的不同：
- Original Tracer: Theory of Mind (心理状态追踪)
- Preference Tracer: User Preference Learning (偏好学习)

但底层的粒子滤波算法和假设管理机制完全一致！🎉

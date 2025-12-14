# PRISM Preference Tracing - File Index

## 🎯 开始这里

| 文件 | 描述 | 优先级 |
|------|------|--------|
| `QUICKSTART_CN.md` | 中文快速开始指南 | ⭐⭐⭐ |
| `test_setup.py` | 环境验证脚本 - 先运行这个 | ⭐⭐⭐ |
| `demo_logging.py` | **日志演示脚本 - 查看详细追踪过程** | ⭐⭐⭐ |
| `run_prism_pipeline.py` | 主运行脚本 | ⭐⭐⭐ |

## 📚 文档

### 中文文档
- **QUICKSTART_CN.md** - 快速开始，包含 tracer 逻辑说明和使用示例
- **LOGGING_GUIDE.md** - **日志功能完整指南，展示每一步的追踪过程**
- **IMPLEMENTATION_SUMMARY.md** - 详细实现总结，核心算法解释
- **PROJECT_SUMMARY.md** - 完整项目文档，包括所有细节

### 英文文档
- **PRISM_README.md** - Complete English documentation

## 💻 核心代码

### 主要模块
```
preference_tracer.py          (350 行)
├── PreferenceTracer 类
│   ├── initialize_hypotheses()     # 初始化假设
│   ├── propagate_hypotheses()      # 传播假设
│   ├── weigh_hypotheses()          # 加权假设
│   ├── summarize_hypotheses()      # 汇总画像
│   ├── evaluate_generation()       # 评估生成质量
│   ├── predict_choice()            # 预测用户选择
│   └── trace_user_preferences()    # 完整追踪流程
└── run_preference_tracing()        # 主运行函数
```

```
survey_evaluator.py           (150 行)
├── SurveyEvaluator 类
│   ├── load_survey_data()          # 加载问卷数据
│   ├── extract_survey_profile()    # 提取问卷画像
│   ├── evaluate_profile_alignment()# 评估对齐度
│   └── evaluate_all_profiles()     # 批量评估
└── run_survey_evaluation()         # 主运行函数
```

```
visualize_results.py          (150 行)
├── plot_learning_curves()          # 学习曲线
├── plot_survey_alignment()         # 问卷对齐图
├── plot_user_trajectories()        # 用户轨迹
└── generate_all_plots()            # 生成所有图表
```

### 运行脚本
```
run_prism_pipeline.py         (100 行)
└── 主流程控制
    ├── Stage 1: 偏好追踪
    ├── Stage 2: 问卷评估
    ├── Stage 3: 可视化
    └── 参数: --print, --save-detailed-traces (详细日志)
```

```
demo_logging.py               (60 行)
└── 日志演示脚本
    ├── 预配置详细日志输出
    ├── 追踪 1-2 个用户
    └── 展示完整追踪过程
```

```
batch_runner.py               (200 行)
└── 批量实验运行器
    ├── 单个实验运行
    ├── 消融实验运行
    └── 结果汇总
```

## 🧪 测试与配置

```
test_setup.py                 (120 行)
└── 环境验证
    ├── 测试导入
    ├── 测试数据集加载
    └── 测试假设创建
```

```
configs.yaml
└── 实验配置模板
    ├── quick_test           # 快速测试
    ├── standard             # 标准实验
    ├── high_quality         # 高质量实验
    ├── ablation_hypotheses  # 假设数消融
    ├── ablation_models      # 模型消融
    └── full_dataset         # 完整数据集
```

```
requirements_prism.txt
└── Python 依赖列表
```

## 📊 输出文件

运行后会生成在 `preference_results/` 目录:

```
preference_results/
├── preference_tracing_results_{run_id}.json    # 详细结果
│   └── 每个用户的每轮结果，包括：
│       - 用户画像
│       - 假设和权重
│       - 生成分数
│       - 预测准确率
│
├── preference_tracing_summary_{run_id}.json    # 统计摘要
│   └── 按轮次聚合的指标：
│       - 生成分数 (mean, std, CI)
│       - 预测准确率 (mean, std, CI)
│
├── survey_evaluation_{run_id}.json             # 问卷对齐评估
│   └── 每个用户的对齐分数：
│       - 沟通风格
│       - 价值观对齐
│       - 偏好一致性
│       - 总体准确度
│
└── plots/                                      # 可视化图表
    ├── learning_curves.png                    # 学习曲线
    ├── user_trajectories.png                  # 用户轨迹
    └── survey_alignment.png                   # 问卷对齐
```

## 🚀 使用流程

### 最简单的使用方式（查看详细日志）

```bash
# 1. 验证环境
python test_setup.py

# 2. 运行日志演示（推荐首次使用）
python demo_logging.py

# 3. 查看完整的追踪过程
# 终端会显示：
#   - 初始化假设
#   - 传播假设
#   - 加权假设
#   - 用户画像总结
#   - 每轮评估结果
```

### 标准使用方式

```bash
# 运行小规模测试（有日志）
python run_prism_pipeline.py --stage all --n-users 5 --print --run-id test

# 运行标准实验（无日志，更快）
python run_prism_pipeline.py --stage all --n-users 20 --run-id standard

# 查看结果
ls preference_results/
```

### 使用配置文件

```bash
# 运行预定义配置
python batch_runner.py --experiments quick_test

# 运行消融实验
python batch_runner.py --ablation-studies ablation_hypotheses
```

### 分阶段运行

```bash
# 只运行追踪
python run_prism_pipeline.py --stage trace --n-users 10 --run-id exp1

# 只运行评估
python run_prism_pipeline.py --stage evaluate --run-id exp1

# 只生成图表
python run_prism_pipeline.py --stage visualize --run-id exp1
```

## 📖 推荐阅读顺序

### 新手 (第一次使用)
1. `QUICKSTART_CN.md` - 了解系统和 tracer 逻辑
2. 运行 `python test_setup.py` - 验证环境
3. 运行 `python run_prism_pipeline.py --stage all --n-users 5 --run-id test`
4. 查看 `preference_results/` 中的输出

### 研究者 (深入理解)
1. `PROJECT_SUMMARY.md` - 完整系统文档
2. `IMPLEMENTATION_SUMMARY.md` - 实现细节
3. 阅读 `preference_tracer.py` 源码
4. 阅读 `thought-tracing/tracer.py` 源码 (原始框架)

### 开发者 (扩展系统)
1. `IMPLEMENTATION_SUMMARY.md` - 理解架构
2. `preference_tracer.py` - 核心实现
3. 修改 `PreferenceTracer` 类的方法
4. 参考 "扩展与改进" 章节

## 🔍 关键概念速查

### 日志功能（新增）
```bash
# 启用详细日志
--print                    # 终端显示彩色日志
--save-detailed-traces     # 保存每个用户的详细追踪文件

# 演示脚本
python demo_logging.py     # 预配置的日志演示
```

**日志显示内容**：
- 🎯 初始化假设生成
- 🔄 假设传播过程
- ⚖️ 假设加权（显示每个假设的权重和分数）
- 📊 重采样/Rejuvenate 触发
- 📝 用户画像总结
- ✅ 每轮评估结果（生成分数、预测准确率）

详见 `LOGGING_GUIDE.md`

### Tracer 核心流程
```
Initialize → Propagate → Weigh → Resample → Summarize
```

### 输入输出映射
```
Input:  对话历史 + 候选回复 + 用户选择
        ↓
Process: 维护 n 个偏好假设，逐轮更新权重
        ↓
Output: 用户偏好画像 + 性能指标
```

### 评估指标
- **Generation Score** (0-1): 生成回复与选择的相似度
- **Prediction Accuracy** (0-1): 是否正确预测用户选择
- **Survey Alignment** (1-10): 与问卷数据的对齐度

### 关键参数
- `n_hypotheses`: 维护的假设数量 (推荐 4-6)
- `n_users`: 处理的用户数量 (测试 5-10，完整 20-100)
- `tracing_model`: 追踪模型 (推荐 gpt-4o-mini)
- `eval_model`: 评估模型 (推荐 gpt-4o-mini)

## 🆘 故障排除

### 导入错误
```bash
# 确保在正确目录
cd /path/to/preference-tracing

# 检查 thought-tracing 子目录存在
ls thought-tracing/

# 运行验证脚本
python test_setup.py
```

### API 错误
```bash
# 检查 API key
echo $OPENAI_API_KEY

# 设置 API key
export OPENAI_API_KEY="your-key-here"
```

### 数据集加载失败
```bash
# 手动测试
python -c "from datasets import load_dataset; load_dataset('HannahRoseKirk/prism-alignment')"
```

## 📞 支持

- **代码问题**: 查看 `IMPLEMENTATION_SUMMARY.md`
- **使用问题**: 查看 `QUICKSTART_CN.md`
- **概念问题**: 查看 `PROJECT_SUMMARY.md`
- **原始 Tracer**: 查看 `thought-tracing/` 目录

## 🎉 完整文件清单

### 新创建的文件 (9 个)
- [x] `preference_tracer.py` - 核心追踪引擎
- [x] `survey_evaluator.py` - 问卷评估
- [x] `visualize_results.py` - 可视化
- [x] `run_prism_pipeline.py` - 主运行脚本
- [x] `batch_runner.py` - 批量运行器
- [x] `test_setup.py` - 环境验证
- [x] `configs.yaml` - 配置模板
- [x] `requirements_prism.txt` - 依赖列表
- [x] `.gitignore_prism` - Git 忽略规则

### 文档文件 (5 个)
- [x] `QUICKSTART_CN.md` - 快速开始 (中文)
- [x] `PRISM_README.md` - 完整文档 (英文)
- [x] `IMPLEMENTATION_SUMMARY.md` - 实现总结
- [x] `PROJECT_SUMMARY.md` - 项目总文档
- [x] `FILE_INDEX.md` - 本文件

**总计**: 14 个新文件，约 2000+ 行代码和文档

---

## ✅ 准备就绪检查清单

在开始实验前，确认：

- [ ] 已阅读 `QUICKSTART_CN.md`
- [ ] 已运行 `python test_setup.py` 并通过所有测试
- [ ] 已设置 `OPENAI_API_KEY` 环境变量
- [ ] 已安装 `requirements_prism.txt` 中的依赖
- [ ] 已理解输出文件结构
- [ ] 已选择合适的配置 (从 `configs.yaml`)

全部完成后，运行：
```bash
python run_prism_pipeline.py --stage all --n-users 5 --run-id first_test
```

祝实验顺利！🚀

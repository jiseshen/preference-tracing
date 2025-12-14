#!/usr/bin/env python3
"""
测试检查点和数据管理系统
"""
import os
import sys
import json
import argparse

# 测试检查点管理器
def test_checkpoint_manager():
    print("\n" + "="*60)
    print("测试检查点管理器")
    print("="*60)
    
    from checkpoint_manager import CheckpointManager
    
    # 创建临时检查点
    test_dir = 'test_checkpoints'
    os.makedirs(test_dir, exist_ok=True)
    
    cp = CheckpointManager(test_dir, 'test_run')
    
    # 测试标记用户
    print("\n1. 标记用户开始处理...")
    cp.mark_user_started('user_001')
    print(f"   ✓ 用户状态: {cp.checkpoint['user_status']['user_001']}")
    
    print("\n2. 标记用户完成...")
    cp.mark_user_completed('user_001', turns_completed=5)
    print(f"   ✓ 已完成用户: {cp.get_completed_users()}")
    
    print("\n3. 标记用户失败...")
    cp.mark_user_failed('user_002', error='Test error')
    print(f"   ✓ 失败用户: {cp.get_failed_users()}")
    
    print("\n4. 获取待处理用户...")
    all_users = ['user_001', 'user_002', 'user_003', 'user_004']
    pending = cp.get_pending_users(all_users)
    print(f"   ✓ 待处理用户: {pending}")
    
    print("\n5. 进度摘要...")
    summary = cp.get_progress_summary()
    print(f"   ✓ 已完成: {summary['completed']}")
    print(f"   ✓ 失败: {summary['failed']}")
    print(f"   ✓ 最后更新: {summary['last_updated']}")
    
    # 清理
    import shutil
    shutil.rmtree(test_dir)
    print("\n✅ 检查点管理器测试通过！")


# 测试数据管理器
def test_data_manager():
    print("\n" + "="*60)
    print("测试数据管理器")
    print("="*60)
    
    from data_manager import DataManager
    
    # 创建临时数据管理器
    test_dir = 'test_data'
    dm = DataManager(test_dir, 'test_run', save_logs=True)
    
    print("\n1. 开始用户日志记录...")
    dm.start_user_logging('user_001')
    dm.log("测试日志消息 1")
    dm.log("测试日志消息 2")
    print("   ✓ 日志已记录")
    
    print("\n2. 保存轮次指标...")
    metrics = {
        'gen_score': 0.85,
        'prediction_correct': True,
        'ess': 3.42,
        'text_diversity': 0.76
    }
    dm.save_turn_metrics('user_001', 0, metrics)
    print("   ✓ 指标已保存")
    
    print("\n3. 保存最终对齐分数...")
    dm.save_final_alignment('user_001', 0.88, {'match': True})
    print("   ✓ 对齐分数已保存")
    
    print("\n4. 保存用户日志...")
    dm.save_user_log()
    log_file = os.path.join(dm.logs_dir, 'user_user_001.log')
    assert os.path.exists(log_file), "日志文件未创建"
    print(f"   ✓ 日志文件: {log_file}")
    
    print("\n5. 读取日志内容...")
    with open(log_file, 'r', encoding='utf-8') as f:
        log_content = f.read()
    print(f"   ✓ 日志内容预览:\n{log_content[:200]}...")
    
    print("\n6. 保存用户追踪数据...")
    trace_data = {
        'user_id': 'user_001',
        'turn_results': [{'turn': 0, 'score': 0.85}]
    }
    dm.save_user_trace('user_001', trace_data)
    trace_file = os.path.join(dm.traces_dir, 'trace_user_user_001.json')
    assert os.path.exists(trace_file), "追踪文件未创建"
    print(f"   ✓ 追踪文件: {trace_file}")
    
    print("\n7. 获取用户指标...")
    user_metrics = dm.get_user_metrics('user_001')
    print(f"   ✓ 指标数据: {json.dumps(user_metrics, indent=2)[:200]}...")
    
    print("\n8. 获取汇总统计...")
    summary = dm.get_summary_statistics()
    print(f"   ✓ 总用户数: {summary['total_users']}")
    print(f"   ✓ 已完成用户: {summary['completed_users']}")
    print(f"   ✓ 平均对齐分数: {summary['average_alignment_score']:.3f}")
    
    print("\n9. 导出汇总报告...")
    summary_file = dm.export_summary()
    assert os.path.exists(summary_file), "汇总文件未创建"
    print(f"   ✓ 汇总文件: {summary_file}")
    
    # 清理
    import shutil
    shutil.rmtree(test_dir)
    print("\n✅ 数据管理器测试通过！")


# 测试目录结构
def test_directory_structure():
    print("\n" + "="*60)
    print("测试目录结构")
    print("="*60)
    
    from data_manager import DataManager
    from checkpoint_manager import CheckpointManager
    
    test_dir = 'test_structure'
    run_id = 'test_run'
    
    # 创建管理器
    dm = DataManager(test_dir, run_id, save_logs=True)
    cp = CheckpointManager(os.path.join(test_dir, 'checkpoints'), run_id)
    
    # 模拟一些操作
    dm.start_user_logging('user_001')
    dm.log("Test message")
    dm.save_turn_metrics('user_001', 0, {'gen_score': 0.85})
    dm.save_user_log()
    dm.save_user_trace('user_001', {'user_id': 'user_001'})
    
    cp.mark_user_completed('user_001', 3)
    
    print("\n预期目录结构:")
    print(f"""
{test_dir}/
├── checkpoints/
│   └── checkpoint_{run_id}.json
├── logs/
│   └── {run_id}/
│       └── user_user_001.log
├── metrics/
│   └── {run_id}/
│       └── all_metrics.json
└── traces/
    └── {run_id}/
        └── trace_user_user_001.json
    """)
    
    print("\n实际创建的文件:")
    for root, dirs, files in os.walk(test_dir):
        level = root.replace(test_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')
    
    # 验证文件存在
    expected_files = [
        os.path.join(test_dir, 'checkpoints', f'checkpoint_{run_id}.json'),
        os.path.join(test_dir, 'logs', run_id, 'user_user_001.log'),
        os.path.join(test_dir, 'metrics', run_id, 'all_metrics.json'),
        os.path.join(test_dir, 'traces', run_id, 'trace_user_user_001.json'),
    ]
    
    print("\n验证文件...")
    all_exist = True
    for file_path in expected_files:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"   {status} {file_path}")
        all_exist = all_exist and exists
    
    # 清理
    import shutil
    shutil.rmtree(test_dir)
    
    if all_exist:
        print("\n✅ 目录结构测试通过！")
    else:
        print("\n❌ 目录结构测试失败！")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='测试检查点和数据管理系统')
    parser.add_argument('--test', type=str, choices=['checkpoint', 'data', 'structure', 'all'],
                       default='all', help='要运行的测试')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("检查点和数据管理系统测试")
    print("="*60)
    
    try:
        if args.test in ['checkpoint', 'all']:
            test_checkpoint_manager()
        
        if args.test in ['data', 'all']:
            test_data_manager()
        
        if args.test in ['structure', 'all']:
            test_directory_structure()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！")
        print("="*60)
        print("\n可以开始使用以下命令运行实际追踪：")
        print("  python run_prism_pipeline.py --stage trace --n-users 10 --print --resume")
        print("\n查看完整使用指南：")
        print("  cat CHECKPOINT_AND_DATA_GUIDE.md")
        print()
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

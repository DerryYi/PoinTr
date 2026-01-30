#!/usr/bin/env python3
"""
批量分析多个PLY样本
"""

import os
import glob
import pandas as pd
from ply_point_cloud_analyzer import PLYPointCloudAnalyzer

def batch_analyze_ply(data_dir, output_base_dir='./batch_analysis'):
    """
    批量分析目录中的所有PLY样本
    
    目录结构:
    data_dir/
    ├── partial/
    │   ├── sample1.ply
    │   ├── sample2.ply
    │   └── ...
    ├── complete/
    │   ├── sample1.ply
    │   ├── sample2.ply
    │   └── ...
    └── predicted/
        ├── sample1.ply
        ├── sample2.ply
        └── ...
    """
    
    # 查找所有样本
    partial_dir = os.path.join(data_dir, 'partial')
    complete_dir = os.path.join(data_dir, 'complete')
    pred_dir = os.path.join(data_dir, 'predicted')
    
    partial_files = sorted(glob.glob(os.path.join(partial_dir, '*.ply')))
    
    all_results = []
    
    for partial_file in partial_files:
        sample_name = os.path.splitext(os.path.basename(partial_file))[0]
        complete_file = os.path.join(complete_dir, f"{sample_name}.ply")
        pred_file = os.path.join(pred_dir, f"{sample_name}.ply")
        
        if not os.path.exists(complete_file) or not os.path.exists(pred_file):
            print(f"⚠️ 跳过样本 {sample_name}: 缺少完整或预测文件")
            continue
        
        print(f"\n📊 分析样本: {sample_name}")
        print(f"   缺失: {partial_file}")
        print(f"   完整: {complete_file}")
        print(f"   预测: {pred_file}")
        
        # 创建分析器
        analyzer = PLYPointCloudAnalyzer(
            partial_ply=partial_file,
            complete_ply=complete_file,
            pred_ply=pred_file
        )
        
        # 分析叶片特征
        leaf_features = analyzer.analyze_leaf_specific_features()
        
        # 收集结果
        if 'leaf_features' in leaf_features:
            features = leaf_features['leaf_features']
            
            # 提取关键指标
            result = {
                'sample_name': sample_name,
                'num_points_complete': features.get('complete', {}).get('num_points', 0),
                'num_points_pred': features.get('predicted', {}).get('num_points', 0),
                'thickness_complete': features.get('complete', {}).get('thickness', 0),
                'thickness_pred': features.get('predicted', {}).get('thickness', 0),
                'density_complete': features.get('complete', {}).get('density', 0),
                'density_pred': features.get('predicted', {}).get('density', 0),
            }
            
            # 添加补全误差
            if 'completion_error' in features:
                errors = features['completion_error']
                result.update({
                    'chamfer_l2': errors.get('chamfer_l2', 0),
                    'f_score_0.01': errors.get('f_score_0.01', 0),
                    'surface_discontinuity': errors.get('surface_discontinuity', 0),
                })
            
            all_results.append(result)
            
            # 为每个样本生成详细报告
            sample_output_dir = os.path.join(output_base_dir, sample_name)
            analyzer.generate_report(sample_output_dir)
    
    # 生成汇总报告
    if all_results:
        df = pd.DataFrame(all_results)
        summary_path = os.path.join(output_base_dir, 'summary_statistics.csv')
        df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        # 生成统计摘要
        stats_path = os.path.join(output_base_dir, 'summary_statistics.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("批量分析统计摘要\n")
            f.write("=" * 50 + "\n")
            f.write(f"总样本数: {len(all_results)}\n")
            f.write(f"平均Chamfer L2距离: {df['chamfer_l2'].mean():.6f}\n")
            f.write(f"平均F-Score (0.01): {df['f_score_0.01'].mean():.3f}\n")
            f.write(f"平均表面不连续性: {df['surface_discontinuity'].mean():.3f}\n")
            f.write(f"平均厚度误差: {(df['thickness_pred'] - df['thickness_complete']).abs().mean():.4f}\n")
        
        print(f"\n✅ 批量分析完成!")
        print(f"   总样本数: {len(all_results)}")
        print(f"   汇总CSV: {summary_path}")
        print(f"   统计摘要: {stats_path}")

if __name__ == "__main__":
    # 修改为你的数据目录
    data_directory = "/path/to/your/ply/data"
    batch_analyze_ply(data_directory)
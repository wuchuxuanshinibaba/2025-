# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import subprocess
import csv
import random  # 添加random模块导入

def get_true_labels(file_path, sample_ratio=1.0):
    """读取验证集的真实标签，支持数据采样
    sample_ratio: 采样比例，范围0-1，例如0.1表示使用10%的数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            all_data = [{'content': row[1], 'label': 1 if row[6].lower() == 'real' else 0} 
                       for row in reader if len(row) >= 7]
            
            # 计算采样数量并随机采样
            sample_size = int(len(all_data) * sample_ratio)
            sampled_data = random.sample(all_data, sample_size)
            
            print(f"总数据量：{len(all_data)}条")
            print(f"采样数据量：{sample_size}条 ({sample_ratio*100}%)")
            
            return sampled_data
    except Exception as e:
        print(f"读取真实标签时出错：{e}")
        return []

def getNewsClass(content):
    """使用ollama命令行调用gemma3判断新闻真伪"""
    prompt = f"""你是一个专业的新闻真伪鉴别专家。请判断以下新闻是真新闻还是假新闻。
只需回答数字：1表示真新闻，0表示假新闻。
新闻内容：{content[:500]}
"""
    for attempt in range(3):
        try:
            result = subprocess.run(
                ['ollama', 'run', 'gemma3', prompt],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode == 0 and result.stdout:
                # 从输出中提取第一个出现的0或1
                for char in result.stdout:
                    if char in ['0', '1']:
                        return int(char)
            time.sleep(0.5)
        except Exception as e:
            print(f"调用模型出错: {e}")
            time.sleep(0.5)
    return 1  # 默认返回1

def getSentiment(content):
    """使用ollama命令行调用gemma3进行情感分析"""
    prompt = f"""作为一个专业的情感分析专家，请分析以下文本的情感倾向。
只需回答：积极、消极 或 中性。
文本内容：{content[:500]}
"""
    for attempt in range(3):
        try:
            result = r(
                ['ollama', 'run', 'gemma3', prompt],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode == 0 and result.stdout:
                response = result.stdout.lower().strip()
                if '积极' in response:
                    return '积极'
                elif '消极' in response:
                    return '消极'
                elif '中性' in response:
                    return '中性'
            time.sleep(0.5)
        except Exception as e:
            print(f"情感分析出错: {e}")
            time.sleep(0.5)
    return "中性"  # 默认返回中性

def getNewsClassWithSentiment(content, sentiment):
    """结合情感分析的新闻真伪判断"""
    prompt = f"""作为专业的新闻真伪鉴别专家，请综合分析以下信息：

新闻内容：{content[:500]}
情感倾向：{sentiment}

请考虑以下因素：
1. 新闻的客观性和真实性
2. 情感表达是否合理
3. 内容与情感是否匹配

只需回答数字：1表示真新闻，0表示假新闻。
"""
    for attempt in range(3):
        try:
            result = subprocess.run(
                ['ollama', 'run', 'gemma3', prompt],
                capture_output=True,
                text=True,
                encoding='utf-8',
                check=True  # 添加check参数以便捕获错误
            )
            
            if result.stdout:
                for char in result.stdout:
                    if char in ['0', '1']:
                        return int(char)
            time.sleep(1)
        except subprocess.CalledProcessError as e:
            print(f"调用模型失败: {e}")
            print(f"错误输出: {e.stderr}")
            time.sleep(2)  # 增加等待时间
        except FileNotFoundError:
            print("错误: 找不到ollama程序，请确保已正确安装并添加到系统路径")
            return 1
        except Exception as e:
            print(f"调用模型出错: {e}")
            time.sleep(2)
    return 1

def calculate_accuracy_metrics(results, true_labels):
    """计算各项准确率指标"""
    total = len(results)
    total_correct = 0
    true_correct = 0
    fake_correct = 0
    total_true = 0
    total_fake = 0
    
    for result, true_item in zip(results, true_labels):
        true_label = true_item['label']
        if true_label == 1:
            total_true += 1
        else:
            total_fake += 1
            
        if result['predicted'] == true_label:
            total_correct += 1
            if true_label == 1:
                true_correct += 1
            else:
                fake_correct += 1
    
    return {
        'overall_accuracy': (total_correct / total) * 100 if total > 0 else 0,
        'true_accuracy': (true_correct / total_true) * 100 if total_true > 0 else 0,
        'fake_accuracy': (fake_correct / total_fake) * 100 if total_fake > 0 else 0
    }

def main():
    validation_path = os.path.join(
        os.path.expanduser("~"),
        "Desktop",
        "期末大作业",
        "验证集.csv"
    )

    file_path = sys.argv[1] if len(sys.argv) > 1 else validation_path
    print(f"加载验证数据路径：{file_path}")

    try:
        # 设置采样比例，可以通过命令行参数传入
        sample_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1  
        print(f"使用数据采样比例: {sample_ratio*100}%")
        
        true_labels = get_true_labels(file_path, sample_ratio)
        if not true_labels:
            print("没有找到任何数据。")
            return
            
        print(f"\n开始处理验证集数据，共 {len(true_labels)} 条新闻...")
        
        results = []  # 合并后的结果列表
        
        for i, item in enumerate(true_labels, 1):
            content = item['content']
            true_label = item['label']
            
            print(f"\n处理第 {i}/{len(true_labels)} 条新闻...")
            print(f"内容预览: {content[:100]}...")
            
            # 基础方法预测
            basic_prediction = getNewsClass(content)
            
            # 情感分析
            sentiment = getSentiment(content)
            print(f"情感分析: {sentiment}")
            
            # 结合情感分析的预测
            sentiment_prediction = getNewsClassWithSentiment(content, sentiment)
            
            print(f"真实标签: {true_label}")
            print(f"基础预测: {basic_prediction}")
            print(f"结合情感预测: {sentiment_prediction}")
            
            # 合并保存所有结果
            results.append({
                'content': content,
                'true_label': true_label,
                'basic_prediction': basic_prediction,
                'sentiment_prediction': sentiment_prediction,
                'sentiment': sentiment
            })
                       
        # 修改计算准确率的调用
        basic_metrics = calculate_accuracy_metrics(
            [{'predicted': r['basic_prediction']} for r in results], 
            true_labels
        )
        sentiment_metrics = calculate_accuracy_metrics(
            [{'predicted': r['sentiment_prediction']} for r in results], 
            true_labels
        )
        
        # 修改情感统计
        sentiment_stats = {
            '积极': sum(1 for r in results if r['sentiment'] == '积极'),
            '消极': sum(1 for r in results if r['sentiment'] == '消极'),
            '中性': sum(1 for r in results if r['sentiment'] == '中性')
        }
        
        # 计算准确率提升
        accuracy_improvement = sentiment_metrics['overall_accuracy'] - basic_metrics['overall_accuracy']
        
        print("\n=== 准确率对比 ===")
        print("\n基础方法准确率：")
        print(f"整体准确率: {basic_metrics['overall_accuracy']:.2f}%")
        print(f"真新闻准确率: {basic_metrics['true_accuracy']:.2f}%")
        print(f"假新闻准确率: {basic_metrics['fake_accuracy']:.2f}%")
        
        print("\n结合情感分析方法准确率：")
        print(f"整体准确率: {sentiment_metrics['overall_accuracy']:.2f}%")
        print(f"真新闻准确率: {sentiment_metrics['true_accuracy']:.2f}%")
        print(f"假新闻准确率: {sentiment_metrics['fake_accuracy']:.2f}%")
        
        print(f"\n准确率提升: {accuracy_improvement:.2f}%")
        
        # 保存最终结果
        final_results = {
            'Result analysis': {
                'results': results,
                'basic_method': basic_metrics,
                'sentiment_method': sentiment_metrics,
                'improvement': accuracy_improvement
            },
            'sentiment_distribution': sentiment_stats
        }
        
        with open("第一问Result analysis.json", 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
    
        print("\n详细结果已保存到 第一问Result analysis.json")
        
    except Exception as e:
        print(f"错误：{e}")
        return

if __name__ == "__main__":
    main()

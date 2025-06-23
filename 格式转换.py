import csv

def convert_txt_to_csv(input_file, output_file, swap_columns=False):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # 获取原始表头并按tab分割
            headers = lines[0].strip().split('\t')
            
        with open(output_file, 'w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')  # 明确指定使用逗号作为分隔符
            
            # 如果需要交换列顺序
            if swap_columns:
                # 交换表头中 username 和 image_id 的位置
                headers[3], headers[4] = headers[4], headers[3]
            writer.writerow(headers)  # 写入分割后的表头
            
            # 从第二行开始处理数据
            for line in lines[1:]:  # 跳过一个表头行
                try:
                    # 先按tab分割主要字段
                    main_parts = line.strip().split('\t')
                    if len(main_parts) >= 7:
                        if swap_columns:
                            # 交换数据中 username 和 image_id 的位置
                            main_parts[3], main_parts[4] = main_parts[4], main_parts[3]
                        writer.writerow(main_parts[:7])
                    else:
                        print(f"警告: 格式不匹配，跳过行: {line.strip()}")
                        
                except Exception as e:
                    print(f"处理行时出错: {line.strip()}\n错误: {e}")
                    continue
                    
        print(f"成功生成 CSV 文件: {output_file}")
        
    except Exception as e:
        print(f"错误: {e}")

# 转换文件 - posts.csv 需要交换列顺序
convert_txt_to_csv(
    r'c:\Users\wcx\Desktop\期末大作业\twitter_dataset\devset\posts.txt',
    r'c:\Users\wcx\Desktop\期末大作业\posts.csv',
    swap_columns=True
)

# posts_groundtruth.csv 保持原有顺序
convert_txt_to_csv(
    r'c:\Users\wcx\Desktop\期末大作业\twitter_dataset\testset\posts_groundtruth.txt',
    r'c:\Users\wcx\Desktop\期末大作业\posts_groundtruth.csv',
    swap_columns=False
)

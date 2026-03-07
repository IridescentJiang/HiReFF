#!/bin/bash

# 设置输入根目录
input_dir="../vggt_demo/demo_timeconsist"

# 遍历所有子目录
find "$input_dir" -mindepth 1 -maxdepth 1 -type d | while read -r sub_dir; do
    echo "正在处理目录: $sub_dir"

    # 创建结果目录（父级目录下的result目录）
    output_dir="$(dirname "$sub_dir")/result"
    mkdir -p "$output_dir"

    # 执行转换命令（输出到父级result目录）
    python vggt_to_ply.py \
        --image_dir "$sub_dir" \
        --output_dir "$output_dir" \
        --mask_black_bg \
        --prediction_mode 'Pointmap Branch' \
        --conf_threshold 5
        
    echo "已完成处理: $sub_dir"
    echo "--------------------------------"
done

echo "全部目录处理完成！"
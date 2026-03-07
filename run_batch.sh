#!/bin/bash

# 设置输入根目录
input_dir="../vggt_demo/demo_timeconsist_s1a2"


# 执行转换命令（输出到父级result目录）
python vggt_to_ply.py \
    --image_dir "$input_dir" \
    --mask_black_bg \
    --prediction_mode 'Pointmap Branch' \
    --conf_threshold 5 \
    --run_batchs


echo "全部目录处理完成！"
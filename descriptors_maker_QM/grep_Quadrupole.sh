#!/bin/bash

# 定义输出文件
output_file_Q1="result_Q1.out"
output_file_Q2="result_Q2.out"

# 清空输出文件，准备写入
> "$output_file_Q1"
> "$output_file_Q2"

# 循环遍历1到320的值
for i in {1..16}; do
    # 构建文件名
    filename="job_${i}_g16_1.log"

    # 检查文件是否存在
    if [ -f "$filename" ]; then
        # 使用grep命令搜索第一个XX=所在行，并添加文件名前缀后保存到result_Q1.out
        grep -m 1 "XX=" "$filename" | awk '{print "["$0"]"}' | sed "s/\[/[job_${i}_g16.log]: /" >> "$output_file_Q1"

        # 使用grep命令搜索第一个XY=所在行，并添加文件名前缀后保存到result_Q2.out
        grep -m 1 "XY=" "$filename" | awk '{print "["$0"]"}' | sed "s/\[/[job_${i}_g16.log]: /" >> "$output_file_Q2"
    else
        echo "File $filename does not exist." >> "$output_file_Q1"
        echo "File $filename does not exist." >> "$output_file_Q2"
    fi
done

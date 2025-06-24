#!/bin/bash

# 初始化输出文件
result_O1="result_O1.out"
result_O2="result_O2.out"
result_O3="result_O3.out"

# 清空输出文件，准备写入新内容
> "$result_O1"
> "$result_O2"
> "$result_O3"

# 循环变量i的范围
for i in {1..1549}
do
    # 构造日志文件名
    log_file="job_${i}_g16_1.log"
    
    # 检查日志文件是否存在
    if [ ! -f "$log_file" ]; then
        echo "Warning: Log file $log_file does not exist, skipping."
        continue
    fi

    # 搜索关键字并追加到对应的文件
    # 使用 grep 直接匹配整行，并追加到输出文件
    # 使用 awk 来打印特定字段，这里假设关键字后面跟的是空格和值
    awk '/XXX=/ {print $0}' "$log_file" >> "$result_O1"
    echo "File name: $log_file" >> "$result_O1"

    awk '/XXY=/ {print $0}' "$log_file" >> "$result_O2"
    echo "File name: $log_file" >> "$result_O2"

    awk '/YYZ=/ {print $0}' "$log_file" >> "$result_O3"
    echo "File name: $log_file" >> "$result_O3"
done

echo "Extraction completed for all files."

# 脚本执行到这里，你应该根据需要决定是否删除文件
# 如果你想保留文件，可以注释掉或删除下面的命令
# 删除所有包含'result_O'的文件
# rm -f result_O*.out

echo "All 'result_O' files have been processed."

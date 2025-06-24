#!/bin/bash

# 定义一个数组来存储结果
declare -A results

# 遍历特定模式的文件，这里假设i从1开始，并且文件名是连续的
for i in {1..16}; do # 这里假设有100个文件，你可以根据实际情况调整范围
    file="job_i_g16_1.log"
    job_file="${file//\i/$i}" # 替换i为当前的数字

    # 检查文件是否存在
    if [ -f "$job_file" ]; then
        # 使用awk抓取Dipole moment下一行的内容，并存储到关联数组中
        # 这里我们使用i作为键，文件内容作为值
        awk -v key="$i" '$0 ~ /Dipole moment \(field-independent basis, Debye\):/ {getline; print key, $0}' "$job_file" >> temp_result.txt
    fi
done

# 读取临时文件，按照第一列（即i的值）进行排序，并写入result文件
sort -n temp_result.txt > result

# 输出结果
echo "Extraction and sorting completed. Results are in result file."

# 清理临时文件
rm temp_result.txt

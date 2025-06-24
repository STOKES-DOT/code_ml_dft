#!/bin/bash

# 定义起始和结束的i值
start_i=1
end_i=16

# 定义结果文件和对应的匹配模式
result_files=("result_H1.out" "result_H2.out" "result_H3.out" "result_H4.out")
patterns=("XXXX=" "XXXZ=" "ZZZY=" "XXYZ=")

# 清空所有结果文件
for file in "${result_files[@]}"; do
    > "$file"
done

# 循环遍历每个i值
for (( i=start_i; i<=end_i; i++ )); do
    file="job_${i}_g16_1.log"

    # 检查文件是否存在
    if [[ -f "$file" ]]; then
        # 使用awk一次性处理所有模式
        awk -v fname="$file" '
            {
                # 独立检查每个模式，允许一行匹配多个模式
                if ($0 ~ /XXXX=/) { print fname, $0 >> "result_H1.out" }
                if ($0 ~ /XXXZ=/) { print fname, $0 >> "result_H2.out" }
                if ($0 ~ /ZZZY=/) { print fname, $0 >> "result_H3.out" }
                if ($0 ~ /XXYZ=/) { print fname, $0 >> "result_H4.out" }
            }
        ' "$file"
    fi
done

# 对每个结果文件按i值排序
for file in "${result_files[@]}"; do
    sort -t'_' -k2,2n "$file" -o "$file"
done

# 输出完成信息
echo "Extraction and sorting completed. Results are in:"
printf "  - %s\n" "${result_files[@]}"

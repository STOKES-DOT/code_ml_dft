#!/bin/bash

# 设置相对路径（基于脚本位置）
template="test.gjf"
folder_path="../../main/structure"

# 检查模板和文件夹
if [ ! -f "$template" ]; then
    echo "错误：模板文件 $template 不存在。"
    exit 1
fi

if [ ! -d "$folder_path" ]; then
    echo "错误：结构文件夹 $folder_path 不存在。"
    exit 1
fi

# 自动检测xyz文件
xyz_files=("$folder_path"/job_*.xyz)
num_files=${#xyz_files[@]}

if [ $num_files -eq 0 ]; then
    echo "错误：在 $folder_path 中没有找到任何 job_*.xyz 文件。"
    exit 1
fi

echo "找到 $num_files 个结构文件，开始处理..."

# 处理每个文件
for xyz_file in "${xyz_files[@]}"; do
    # 从文件路径中提取任务编号
    file_name=$(basename "$xyz_file")
    task_number=$(echo "$file_name" | grep -oP 'job_\K\d+')
    
    # 基础文件名
    base_name="job_${task_number}_g16"
    gjf_file="${base_name}.gjf"
    
    # 复制模板
    cp "$template" "$gjf_file"
    
    # 更新CHK路径 - 确保两个任务有不同的文件名
    sed -i "
        # 第一个任务的CHK文件
        s|%chk=_n.chk|%chk=${base_name}_n.chk|;
        
        # 第二个任务的CHK文件（使用不同的后缀）
        s|%chk=_o.chk|%chk=${base_name}.chk|;
	s|%ochk=_o.chk|%ochk=${base_name}.chk|
 
    " "$gjf_file"

    # 插入坐标
    {
        # 第一部分：保留第一个作业段的头部
        sed -n '1,/^0 1$/p' "$gjf_file"
        
        # 插入xyz坐标（跳过前两行）
        tail -n +3 "$xyz_file"
        echo ""  # 确保坐标后有空白行
        
        # 第二部分：保留link1之后的设置
        sed -n '/^--link1--/,$p' "$gjf_file"
    } > tmpfile && mv tmpfile "$gjf_file"

    echo "生成: $gjf_file"
    echo "  - 优化任务CHK: ${base_name}.chk"
    echo "  - TDDFT任务CHK: ${base_name}_TDDFT.chk"
done

echo "操作完成。成功生成 $num_files 个高斯输入文件。"

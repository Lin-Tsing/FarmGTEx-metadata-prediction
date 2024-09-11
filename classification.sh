#!/bin/bash
# module load anaconda3/5.0.1-gcc-4.8.5
module load miniconda3/23.9.0

work_path=$1

group_method=$2
num_splits=$3
num_repeats=$4

feature_method=$5
feature_number=$6

classif_method=$7
classic_labels=$8

folder_path=$9

input_X_file=${10}
input_y_file=${11}

input_X_test_file=${12}
# input_y_test_file=${13}



echo ${feature_method} ${feature_number} ${classif_method} ${classic_labels} running!

work_path=/mnt/export/home/nsccwuxi_scau/zhangz2/data/USER/linqing/identification
cd ${work_path}

python ${work_path}/classification.py \
${work_path} \
${group_method} \
${num_splits} \
${num_repeats} \
${feature_method} \
${feature_number} \
${classif_method} \
${classic_labels} \
${folder_path} \
${input_X_file} \
${input_y_file} \
${input_X_test_file}



echo ${group_method} ${feature_method} ${feature_number} ${classif_method} ${classic_labels} done!
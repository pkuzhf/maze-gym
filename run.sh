#!/bin/bash
t=`date "+%Y%m%d_%H%M%S"`
task_name=$1.$t

if [ $# -eq 3 ]
then
  height=$2
  width=$3
  python -u main.py $task_name $height $width > ./logs/$task_name.log
else
  python -u main.py $task_name > ./logs/$task_name.log
fi
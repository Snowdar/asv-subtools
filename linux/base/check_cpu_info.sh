#!/bin/bash 
# Reference: https://www.cnblogs.com/emanlee/p/3587571.html.

cpu=$(cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c | sed 's/ \+/ /g' | sed 's/^ //g')
num_cpu=$(cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l)
cores=$(cat /proc/cpuinfo| grep "cpu cores"| uniq | awk '{print $4}')
processor=$(cat /proc/cpuinfo| grep "processor"| wc -l)

echo "CPU: $cpu"
echo "Num: $num_cpu"
echo "CPU Cores: $cores"
echo "Total Processor: $processor"
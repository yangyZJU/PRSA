#!/bin/bash


tasks="Ads Business Code email Ideas SEO Writing Food Health Music Data Fashion Games Language Sports Study Translate Travel"


parallel_jobs=6

current_jobs=0


for task in $tasks
do
    output_dir="log"
    mkdir -p "$output_dir"
    nohup python -u 1_prompt_attention_generation.py --theme "$task" --out "${output_dir}/${task}.txt" > "${output_dir}/${task}.log" 2>&1 &
    current_jobs=$((current_jobs + 1))

    if [ $current_jobs -ge $parallel_jobs ]; then
        wait
        current_jobs=0
    fi
done

wait

echo "All commands executed."

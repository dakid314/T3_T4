#!/bin/bash

# hostname="parateraHCD"
# if [ $2 ]; then
#     hostname=$2
# fi

nowdate=$(date +%Y-%m-%d\ %H:%M:%S)
curl -s -o /dev/zero -H "t: Job $1" -d "Start.
$nowdate" ntfy.sh/georgezhao;

start=$(date +%s)
while :
do
    if /usr/bin/squeue | grep -q $1;
    then
        continue
    else 
        end=$(date +%s);
        duration=$((end - start));
        time_len="Spent Time: $((duration / 3600))h $((duration % 3600 / 60))m $((duration % 60))s"
        echo "cat ../*$1*" >> $HOME/.bash_history
        curl -s -o /dev/zero -H "t: Job $1" -d "Done.
$time_len" ntfy.sh/georgezhao ;
        break;
    fi
done

# Usage: screen -dmS check ./script/check.sh 415686
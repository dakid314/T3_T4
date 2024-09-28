#!/bin/bash

JobID=`sbatch scripts/feature_selection.sh $@ | sed -e "s/Submitted batch job \([0-9]\{7\}\)/\1/g"`
if [ ! -z $JobID ];
then
    screen -dmS checkjob ./scripts/checkjob.sh $JobID
else
    echo Error.
fi

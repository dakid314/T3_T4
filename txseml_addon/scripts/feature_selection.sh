#!/bin/bash
#SBATCH -p v5_192
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH -o ../TxSEml_Addon-feature_selection-%j.out

date;

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/public1/home/scfa2650/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/public1/home/scfa2650/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/public1/home/scfa2650/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/public1/home/scfa2650/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate TxSEml_Addon

export TEXMFHOME="~/texmf"
export n_jobs=48

export TYPE_TO_ANALYSIS=$1

export FEATURE_PROB=0.2
if [ $2 ]; then
    export FEATURE_PROB=$2
fi

export SAVE_MODEL=0
export SEARCH_MODE="Bayes"

export PYTHONWARNINGS="ignore"

python3 -W ignore -u src/libfeatureselection

date;
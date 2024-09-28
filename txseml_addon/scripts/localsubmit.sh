#!/bin/bash
date;

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.10
export WORKON_HOME=$HOME/.pyvirtualenvs
. /usr/share/virtualenvwrapper/virtualenvwrapper.sh
workon TxSEml_Backend

export n_jobs=30

export TYPE_TO_ANALYSIS=$1
export PYTHONWARNINGS="ignore"
export FEATURE_PROB=0.5
export SAVE_MODEL=0
export SEARCH_MODE="Onehot"

python3 -W ignore -u src/libfeatureselection

date;
curl -s -o /dev/zero -H "t: Job T$1" -d "End.
$nowdate" ntfy.sh/georgezhao;
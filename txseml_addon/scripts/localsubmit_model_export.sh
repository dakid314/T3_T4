#!/bin/bash

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.10
export WORKON_HOME=$HOME/.pyvirtualenvs
. /usr/local/bin/virtualenvwrapper.sh
workon TxSEml_Backend

export n_jobs=30

export TYPE_TO_ANALYSIS=$1
export PYTHONWARNINGS="ignore"
export FEATURE_PROB=0.5
export SAVE_MODEL=1
export SEARCH_MODE="Specification"

python3 -W ignore -u src/libfeatureselection
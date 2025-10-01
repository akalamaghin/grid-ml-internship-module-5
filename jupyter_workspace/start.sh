#!/bin/bash
set -e

export $(grep -v '^#' .env | xargs)

source /jupyter_workspace/.venv/bin/activate

pip install -r /jupyter_workspace/requirements.txt

jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="$JN_PASS"
#!/bin/bash

# 当前脚本路径
SCRIPT_PATH=$(dirname "$(readlink -f "$0")")

# setup安装
cd ${SCRIPT_PATH}/csrc
pip install --no-build-isolation -e .

python3 ${SCRIPT_PATH}/test.py

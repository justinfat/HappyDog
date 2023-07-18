#!/bin/bash

pidfile="$HOME/HappyDog/.lock"
if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo still running
    exit 1
fi
echo $$ > $pidfile

cd ~/HappyDog || exit

export PYTHONPATH=.

npx electron main/GUI_controller/electron/ &
pyenv/bin/python3 main/main.py

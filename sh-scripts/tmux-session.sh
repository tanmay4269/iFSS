#!/bin/bash

SESSION="iFSS"

tmux has-session -t $SESSION 2>/dev/null

if [ $? != 0 ]; then
    tmux new-session -d -s $SESSION -n $SESSION

    tmux new-window -t $SESSION:1 -n "watch"
    tmux send-keys -t ${SESSION}:1 "htop" C-m
    tmux split-window -h -t ${SESSION}:1
    tmux send-keys -t ${SESSION}:1.2 "watch nvidia-smi" C-m

    tmux new-window -t $SESSION:2 -n "git"
    tmux send-keys -t ${SESSION}:2 "watch git status" C-m
    tmux split-window -h -t ${SESSION}:2
    tmux send-keys -t ${SESSION}:2.2 "git branch" C-m

    tmux new-window -t $SESSION:3 -n "tblog"
    tmux send-keys -t ${SESSION}:3 "tensorboard --logdir /home/tvg/Projects/iFSS/experiments" C-m

    tmux new-window -t $SESSION:4 -n "bash"
    tmux send-keys -t ${SESSION}:4 "conda activate iFSS" C-m

    tmux select-window -t $SESSION:2
fi

tmux attach-session -t $SESSION

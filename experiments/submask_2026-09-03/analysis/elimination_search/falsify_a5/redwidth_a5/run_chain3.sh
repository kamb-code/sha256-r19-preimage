#!/bin/bash
cd /tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/falsify/redwidth_a5
until [ -f exhaust_famA_w8.json ]; do sleep 20; done
pkill -f "^/bin/bash ./run_chain2.sh"
sleep 5
while pgrep -f "^python3 exhaust.py rand 8" > /dev/null; do sleep 20; done
python3 exhaust.py famB 8            > exhaust_famB_w8.log 2>&1
python3 exhaust.py rand 12 262144    > exhaust_rand_w12.log 2>&1
python3 exhaust.py kernrand 12 262144 > exhaust_kernrand_w12.log 2>&1
python3 exhaust.py rand 6 16777216   > exhaust_rand_w6.log 2>&1
echo ALLDONE > run_chain3.done

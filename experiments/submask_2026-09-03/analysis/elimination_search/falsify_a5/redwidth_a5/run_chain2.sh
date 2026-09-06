#!/bin/bash
cd /tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/falsify/redwidth_a5
while pgrep -f "^python3 exhaust.py lit 5" > /dev/null; do sleep 15; done
python3 exhaust.py famA 8         > exhaust_famA_w8.log 2>&1
python3 exhaust.py rand 8 2097152 > exhaust_rand_w8.log 2>&1
python3 exhaust.py famB 8         > exhaust_famB_w8.log 2>&1
python3 exhaust.py rand 12 262144 > exhaust_rand_w12.log 2>&1
python3 exhaust.py kernrand 12 1048576 > exhaust_kernrand_w12.log 2>&1
python3 exhaust.py lit 6          > exhaust_lit_w6.log 2>&1
echo ALLDONE > run_chain2.done

#!/bin/bash
cd /tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/elim/a10
until [ -f chain.done ]; do sleep 20; done
for w in a8 a9; do python3 enum_ctx.py $w 12 > enum_$w.log 2>&1; done
echo DONE > chain3.done

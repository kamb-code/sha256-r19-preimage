#!/bin/bash
cd /tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/elim/a10
for w in a11 a7 a6 a5; do python3 enum_ctx.py $w 12 > enum_$w.log 2>&1; done
echo CHAIN_DONE > chain.done

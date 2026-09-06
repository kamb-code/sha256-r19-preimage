#!/bin/bash
cd /tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/elim/a10
until [ -f chain3.done ]; do sleep 20; done
for w in a7 a6; do python3 enum_words.py $w 12 > words_$w.log 2>&1; done
echo DONE > chain4.done

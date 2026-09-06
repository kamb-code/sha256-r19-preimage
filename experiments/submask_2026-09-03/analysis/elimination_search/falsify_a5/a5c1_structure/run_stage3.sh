#!/bin/bash
cd /tmp/claude-1000/-home-administrator-sha/bc6b4995-3384-44e4-9203-a490e2389533/scratchpad/falsify/a5c1_structure
echo "=== A: best-C1 config (a6=a4, a7=a6, e8=-1, e9=a5+0) ==="
python3 -u stage3_structure.py "{6: ('eq', 4), 7: ('eq', 6), 8: ('sat_r', 4294967295), 9: ('hoff', 0)}" --scan 2 --hist 2>&1 | grep --line-buffered -v "^    [#.+,-]"
echo "=== B: best standard-menu config (a6=~a4, a7=a6, e8=-1, e10=-1, e11=0) ==="
python3 -u stage3_structure.py "{6: ('neq', 4), 7: ('eq', 6), 8: ('sat_r', 4294967295), 10: ('sat_r', 4294967295), 11: ('sat_r', 0)}" --scan 2 --hist 2>&1 | grep --line-buffered -v "^    [#.+,-]"
echo "=== C: a6=a4^2^31, a7=a6, e8=-1, e9=a5 ==="
python3 -u stage3_structure.py "{6: ('xor', 4, 2147483648), 7: ('eq', 6), 8: ('sat_r', 4294967295), 9: ('hoff', 0)}" --scan 1 2>&1 | grep --line-buffered -v "^    [#.+,-]"
echo "=== D: random context (no conditions) ==="
python3 -u stage3_structure.py "{}" --scan 1 2>&1 | grep --line-buffered -v "^    [#.+,-]"
echo STAGE3_DONE

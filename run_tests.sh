#!/bin/sh
# Run the tiny random-weight model tests (no model download; FP32/BF16, a few minutes on an M-series GPU).
# Usage, from the mlx-vlm tree that install.sh overlaid:  ./run_tests.sh [python]
# Each line of tests.txt: name|ENV=VAL ...|script [args]. A test passes iff it exits 0 and prints "PASS".
PY=${1:-python}
ALL31=head,mask,glue,fuse,fusemoe,route,moesort,ctrlbatch,xpass,msgpack,qkvclip,xmask,sconvtail,attnres,mega_ctrl,mega_norm,mega_dsconv,mega_moe,mega_attnres,mega_kvz,mega_post,mega_qsm,mega_sconv2,mega_oproj,mega_kvip,mega_pre,mega_dpar,mega_oproj2,mega_qsm2,mega_rg2,mega_sdpa
pass=0; fail=0; failed=""
LOGDIR=${TMPDIR:-/tmp}/inkling2_fast_tests.$$; mkdir -p "$LOGDIR"
while IFS='|' read -r name envs cmd; do
  case "$name" in ''|'#'*) continue;; esac
  envs=$(echo "$envs" | sed "s/\$ALL31/$ALL31/g")
  if env $envs PYTHONPATH="$PWD" $PY $cmd > "$LOGDIR/$name.log" 2>&1 && grep -q "PASS" "$LOGDIR/$name.log"; then
    pass=$((pass+1)); echo "PASS $name"
  else
    fail=$((fail+1)); failed="$failed $name"; echo "FAIL $name (log: $LOGDIR/$name.log)"; tail -n 5 "$LOGDIR/$name.log"
  fi
done < tests.txt
echo "passed $pass, failed $fail${failed:+ ($failed )}"
[ $fail -eq 0 ]

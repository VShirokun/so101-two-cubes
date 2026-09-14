#!/usr/bin/env bash
# Автотесты софта реального стенда (без железа) + опциональный дымовой прогон на стенде.
#   bash tests/run_real_tests.sh          # только софт, ~1 мин, руку не трогает
#   bash tests/run_real_tests.sh --rig    # плюс стенд: камеры свежие, порт, чтение руки, 1 эпизод сборщика
set -u
cd "$(dirname "$0")/.."
PY=/srv/data/vshirokun/groot-venv/bin/python
export MUJOCO_GL=egl LD_LIBRARY_PATH=/srv/data/vshirokun/ffmpeg7-shim:/srv/data/vshirokun/groot-venv/lib/python3.12/site-packages/av.libs
$PY -m pytest tests/real -q 2>&1 | tail -15
rc=${PIPESTATUS[0]}
if [ "${1:-}" = "--rig" ]; then
  echo "== стенд =="
  for f in /tmp/roboom_cam/latest.jpg /tmp/roboom_wrist/latest.jpg; do
    age=$(( $(date +%s) - $(stat -c %Y $f 2>/dev/null || echo 0) )); [ $age -lt 3 ] && echo "ok   $f свежий" || { echo "FAIL $f старше 3 с"; rc=1; }
  done
  test -w /dev/ttyACM0 && echo "ok   /dev/ttyACM0 доступен" || { echo "FAIL порт руки недоступен"; rc=1; }
  (cd real/arm && $PY -c "from arm_driver import ArmDriver; d=ArmDriver(); q,_=d.read(); print('ok   рука отвечает, z=%.3f' % d.tcp_z(q[:5])[0]); d.close()" 2>&1 | grep -v '^\[W') || rc=1
  (cd real/arm && $PY -u collect_pilot.py --episodes 1 --out ../data/smoke 2>&1 | grep -v '^\[W' | grep -E "эпизод|доводка|серия" | tail -3) || rc=1
fi
exit $rc

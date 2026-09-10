#!/bin/bash
cd "$(dirname "$0")"
exec /opt/anaconda3/envs/lerobot/bin/python -u dashboard.py 2>&1 | tee /tmp/roboom_panel.log

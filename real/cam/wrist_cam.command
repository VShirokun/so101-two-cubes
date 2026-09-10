#!/bin/bash
cd "$(dirname "$0")"
exec /opt/anaconda3/envs/lerobot/bin/python -u cam_server.py --name USB2.0 --width 640 --dir /tmp/roboom_wrist 2>&1 | tee /tmp/roboom_wrist/server.log

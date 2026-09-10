#!/bin/bash
cd "$(dirname "$0")"
exec /opt/anaconda3/envs/lerobot/bin/python -u cam_server.py 2>&1 | tee /tmp/roboom_cam/server.log

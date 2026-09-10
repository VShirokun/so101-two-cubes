#!/bin/bash
ffmpeg -hide_banner -f avfoundation -framerate 30 -video_size 1280x720 -i "HD Pro Webcam C920" -t 6 -f null - > /tmp/roboom_cam/fps_probe.log 2>&1
echo "probe done" >> /tmp/roboom_cam/fps_probe.log

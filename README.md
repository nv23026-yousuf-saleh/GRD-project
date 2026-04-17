# Plithos

Smart safety monitoring for schools.

Plithos is an AI-powered monitoring system built for school environments. It helps staff respond faster by detecting fights, fire hazards, and smoking activity from live camera feeds in real time.

## What it does

- Detects fights and aggressive behavior
- Detects fire and smoke hazards
- Tracks smoking activity
- Shows live camera feeds and alerts in a web dashboard
- Sends alert emails with a CSV report

## Main pages

- Landing page
- Login
- Camera setup
- Dashboard
- All cameras
- Logs
- Settings

## Technology

- Python
- Flask
- YOLO
- OpenCV
- AWS

## Docker

- Build and start: `docker compose up --build`
- Open the site at `http://localhost:5000`
- SQLite data is stored in `./data` through the Docker volume mapping
- On Windows Docker Desktop, direct local webcam indexes such as `0` and `1` may not be available inside the container
- For Docker on Windows, use IP/RTSP camera sources, or run the app natively if you need direct webcam access

## Privacy

Plithos focuses on situations, not identity. It does not use facial recognition.

## Team

- Mohammed Fardan — Team Lead / Cloud / Frontend
- Yousif Alaali — Cloud / Database
- Ali Yasser — Software Developer / AI Integration
- Salman Ashoor — Software Developer / Hardware / R&D

© 2026 Plithos. All rights reserved.

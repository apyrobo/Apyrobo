# APYROBO Demo Environment

Clone the repo and run:

```bash
docker compose -f docker/docker-compose-demo.yml up
```

Then open http://localhost:8000 in your browser.

Three services start automatically: the main orchestration server, a web dashboard, and a mock fleet of 3 robots (turtlebot4, ur5, spot). No ROS or Gazebo required.

# VHS tape — re-render with demos/nav2_gazebo/record.sh (Linux only).
# The sim (gazebo-nav-sim) must already be up and healthy; record.sh does that.
# https://github.com/charmbracelet/vhs

Output demos/nav2_gazebo/demo.gif
Output demos/nav2_gazebo/demo.mp4

Set Shell bash
Set FontSize 14
Set Width 940
Set Height 640
Set Padding 14
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
# Discovery + Nav2 navigation with physics takes a while.
Set WaitTimeout 300s

Type "docker compose -f docker/docker-compose.yml --profile gazebo-nav run --rm gazebo-nav-demo"
Sleep 600ms
Enter

Wait
Sleep 4s

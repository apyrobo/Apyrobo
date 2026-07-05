# VHS tape — re-render with demos/warehouse_robots/record.sh
# https://github.com/charmbracelet/vhs

Output demos/warehouse_robots/demo.gif
Output demos/warehouse_robots/demo.mp4

Set Shell bash
Set FontSize 14
Set Width 940
Set Height 640
Set Padding 14
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set WaitTimeout 120s

Hide
Type "export APYROBO_DEMO_PACE=0.30 && clear"
Enter
Show

Type "python demos/warehouse_robots/demo.py"
Sleep 600ms
Enter

Wait
Sleep 4s

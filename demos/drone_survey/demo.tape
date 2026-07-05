# VHS tape — re-render with demos/drone_survey/record.sh
# https://github.com/charmbracelet/vhs

Output demos/drone_survey/demo.gif
Output demos/drone_survey/demo.mp4

Set Shell bash
Set FontSize 14
Set Width 940
Set Height 640
Set Padding 14
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set WaitTimeout 120s

Hide
Type "export APYROBO_DEMO_PACE=0.45 && clear"
Enter
Show

Type "python demos/drone_survey/demo.py"
Sleep 600ms
Enter

Wait
Sleep 4s

# VHS tape — re-render with demos/humanoid_nlp/record.sh
# https://github.com/charmbracelet/vhs

Output demos/humanoid_nlp/demo.gif
Output demos/humanoid_nlp/demo.mp4

Set Shell bash
Set FontSize 14
Set Width 940
Set Height 640
Set Padding 14
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set WaitTimeout 120s

Hide
Type "export APYROBO_DEMO_PACE=0.40 && clear"
Enter
Show

Type "python demos/humanoid_nlp/demo.py"
Sleep 600ms
Enter

Wait
Sleep 4s

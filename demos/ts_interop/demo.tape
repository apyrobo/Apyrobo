# VHS tape — re-render with demos/ts_interop/record.sh (any OS, mock robot).
# Requires apyrobo installed and the TS client built; record.sh checks both.
# https://github.com/charmbracelet/vhs

Output demos/ts_interop/demo.gif
Output demos/ts_interop/demo.mp4

Set Shell bash
Set FontSize 14
Set Width 940
Set Height 640
Set Padding 14
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set WaitTimeout 120s

Type "node demos/ts_interop/demo.mjs"
Sleep 600ms
Enter

Wait
Sleep 4s

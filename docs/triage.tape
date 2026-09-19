# Recording for docs/assets/triage.gif
#   vhs docs/triage.tape
# Run the demo once first so the model weights are in the local cache,
# otherwise the recording spends its first seconds on a download.

Output docs/assets/triage.gif

Set Shell "bash"
Set FontSize 15
Set Width 1180
Set Height 700
Set Padding 22
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms

Type "python examples/triage_inbox.py"
Sleep 400ms
Enter
Sleep 14s

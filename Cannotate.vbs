' Windows: double-click this in Explorer to start Cannotate with no console window.
'
' Cannotate.bat does the same thing but leaves a black console window open
' behind the GUI. This wrapper runs the same launcher hidden instead.
' If something goes wrong, launch.py still reports it in a dialog box.

Option Explicit

Dim shell, fso, here, pythonCmd, quoted
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

here = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = here

' pythonw.exe runs without allocating a console; plain python.exe would flash
' one open. Fall back through the usual names.
pythonCmd = ""
If FoundOnPath(shell, "pythonw.exe") Then
    pythonCmd = "pythonw.exe"
ElseIf FoundOnPath(shell, "python.exe") Then
    pythonCmd = "python.exe"
ElseIf FoundOnPath(shell, "py.exe") Then
    pythonCmd = "py.exe"
End If

If pythonCmd = "" Then
    MsgBox "No Python interpreter was found." & vbCrLf & vbCrLf & _
           "Install Python 3 from https://www.python.org/downloads/ and tick" & vbCrLf & _
           """Add python.exe to PATH"" in the installer.", _
           vbCritical, "Cannotate - cannot start"
    WScript.Quit 1
End If

quoted = """" & pythonCmd & """ """ & fso.BuildPath(here, "launch.py") & """"
' 0 = hidden window, False = do not wait for it to finish
shell.Run quoted, 0, False

Function FoundOnPath(sh, exeName)
    Dim probe
    On Error Resume Next
    ' "where" exits non-zero when the command is not on PATH.
    probe = sh.Run("cmd /c where " & exeName, 0, True)
    FoundOnPath = (Err.Number = 0 And probe = 0)
    On Error Goto 0
End Function

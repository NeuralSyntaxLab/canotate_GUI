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

' The interpreter setup\install_windows.bat recorded wins: on a PC with no
' system Python -- the machine that setup script exists for -- there is
' nothing on PATH to fall back to.
pythonCmd = ConfiguredPython(fso, here)

' Otherwise: pythonw.exe runs without allocating a console; plain python.exe
' would flash one open. Fall back through the usual names.
If pythonCmd = "" Then
    If FoundOnPath(shell, "pythonw.exe") Then
        pythonCmd = "pythonw.exe"
    ElseIf FoundOnPath(shell, "python.exe") Then
        pythonCmd = "python.exe"
    ElseIf FoundOnPath(shell, "py.exe") Then
        pythonCmd = "py.exe"
    End If
End If

If pythonCmd = "" Then
    MsgBox "Cannotate is not set up on this PC yet." & vbCrLf & vbCrLf & _
           "Open the ""setup"" folder next to this file and double-click" & vbCrLf & _
           "    install_windows.bat" & vbCrLf & vbCrLf & _
           "That installs everything Cannotate needs, including the" & vbCrLf & _
           "neural-network features. It needs no administrator rights.", _
           vbCritical, "Cannotate - not set up yet"
    WScript.Quit 1
End If

quoted = """" & pythonCmd & """ """ & fso.BuildPath(here, "launch.py") & """"
' 0 = hidden window, False = do not wait for it to finish
shell.Run quoted, 0, False

Function ConfiguredPython(fsObj, projectDir)
    ' Read the "python = ..." line out of launcher.cfg, preferring the
    ' console-free pythonw.exe beside it when there is one.
    Dim cfgPath, stream, cfgLine, value, exeDir, pythonw
    ConfiguredPython = ""
    cfgPath = fsObj.BuildPath(projectDir, "launcher.cfg")
    If Not fsObj.FileExists(cfgPath) Then Exit Function

    Set stream = fsObj.OpenTextFile(cfgPath, 1)
    Do Until stream.AtEndOfStream
        cfgLine = Trim(stream.ReadLine)
        If LCase(Left(cfgLine, 6)) = "python" And InStr(cfgLine, "=") > 0 Then
            value = Trim(Mid(cfgLine, InStr(cfgLine, "=") + 1))
            If fsObj.FileExists(value) Then
                exeDir = fsObj.GetParentFolderName(value)
                pythonw = fsObj.BuildPath(exeDir, "pythonw.exe")
                If fsObj.FileExists(pythonw) Then
                    ConfiguredPython = pythonw
                Else
                    ConfiguredPython = value
                End If
            End If
            Exit Do
        End If
    Loop
    stream.Close
End Function

Function FoundOnPath(sh, exeName)
    Dim probe
    On Error Resume Next
    ' "where" exits non-zero when the command is not on PATH.
    probe = sh.Run("cmd /c where " & exeName, 0, True)
    FoundOnPath = (Err.Number = 0 And probe = 0)
    On Error Goto 0
End Function

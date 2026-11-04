' Ejecuta iniciar_streamlit_local.bat de forma 100% invisible (sin ventana negra de cmd)
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
batPath = scriptDir & "\iniciar_streamlit_local.bat"

WshShell.Run """" & batPath & """", 0, False

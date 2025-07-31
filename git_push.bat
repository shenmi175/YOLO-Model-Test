@echo off
REM Enter the folder path to push
set /p folder=Please enter the folder to push (such as src or src  subfolder):
git add "%folder%"
set /p msg=Please enter the commit information:
git commit -m "%msg%"
REM 强制推送到远程
git push -f
pause


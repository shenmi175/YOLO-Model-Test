@echo off
git add .
set /p msg=message:
git commit -m "%msg%"
REM 强制推送本地到远程
git push -f
pause


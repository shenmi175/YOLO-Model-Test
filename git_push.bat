@echo off
REM 自动化 git 推送脚本（适用于 Windows）

REM 添加所有更改
git add .

REM 输入提交信息
set /p msg=message:

REM 提交更改
git commit -m "%msg%"

REM 推送到远程仓库
git push

pause

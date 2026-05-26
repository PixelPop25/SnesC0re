@echo off
powershell -ExecutionPolicy Bypass -File .\build_with_zig.ps1
python .\make_snes_lua.py
echo.
pause
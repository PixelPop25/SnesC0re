@echo off
echo.         powershell -ExecutionPolicy Bypass -File .\build_with_zig.ps1
echo.         python .\make_snes_lua.py
echo.
cmd
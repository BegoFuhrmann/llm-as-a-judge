@echo off
echo Starting RAG-Topic Integration Test...
echo Current directory: %CD%

REM Change to the agentic_system directory
cd /d "\\dnsbego.de\dfsbego\home04\FuhrmannD\Documents\01_Trainee\Master\Thesis\code\agentic_system"

echo New directory: %CD%
echo.

REM Run the test script
python test_simple_integration.py

echo.
echo Test completed. Press any key to exit...
pause > nul

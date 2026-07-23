@echo off
cd /d "%~dp0"
echo [%date% %time%] Starting tool discovery pipeline...
python run_pipeline.py "bioinformatics protein engineering tools" 5
echo [%date% %time%] Pipeline finished.

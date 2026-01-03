import os
import sys
from cli import cli

if __name__ == "__main__":
    print("Starting datasets analysis...")
    
    # Тестовые данные
    for file in [
        "Из статьи.json",
        "Контрпример_1.json",
        "Контрпример_2.json",
        "Магистерский датасет.json",
    ]:
        print(f"Processing file: {file}")
        sys.argv = [
            "data_analysis_pipeline.py",
            "analyze",
            "-m",
            "EABM",
            "-b",
            "local",
            "-f",
            file
        ]
        try:
            cli()
        except SystemExit:
            pass
    
    
    # Синетические данные
    for file in [
        file for file in os.listdir("./synthetic_data") if file.endswith(".json")
    ]:
        print(f"Processing file: {file}")
        sys.argv = [
            "data_analysis_pipeline.py",
            "analyze",
            "-m",
            "EABM",
            "-b",
            "local",
            "-f",
            f"./synthetic_data/{file}",
            "-o",
            "./synthetic_data/results/"
        ]
        try:
            cli()
        except SystemExit:
            pass
    print("All files processed")

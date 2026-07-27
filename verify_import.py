import sys
from pathlib import Path
repo = Path(r'C:\Users\KDT013\Documents\RAG 프로젝트\integrated_final_pipeline')
sys.path.insert(0, str(repo / 'src'))
import main
print('import ok', main.QUIZ_TOPICS)

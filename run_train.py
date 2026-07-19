import sys
import os

# Добавляем путь к проекту
project_path = '/content/semantic_scene_description'
sys.path.insert(0, project_path)
os.chdir(project_path)

# Запускаем train.py
exec(open('train.py').read())

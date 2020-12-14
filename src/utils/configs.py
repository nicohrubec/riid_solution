from pathlib import Path


project_dir = Path.cwd().parent
data_dir = project_dir / 'data'
model_dir = project_dir / 'models'

train_file = data_dir / 'train.csv'
questions_file = data_dir / 'questions.csv'
lectures_file = data_dir / 'lectures.csv'
all_data_file = data_dir / 'train_all.csv'



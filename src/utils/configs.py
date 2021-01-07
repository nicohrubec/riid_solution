from pathlib import Path


project_dir = Path.cwd().parent
data_dir = project_dir / 'data'
model_dir = project_dir / 'models'
oof_dir = project_dir / 'oof'
log_dir = project_dir / 'runs'

train_file = data_dir / 'train.csv'
questions_file = data_dir / 'questions.csv'
lectures_file = data_dir / 'lectures.csv'
all_data_file = data_dir / 'train_all.csv'
all_data_file_pkl = data_dir / 'train_all.pkl'

count_dict_path = data_dir / 'count_dict.pkl'
correct_dict_path = data_dir / 'correct_dict.pkl'
content_dict_path = data_dir / 'content_dict.pkl'
time_dict_path = data_dir / 'time_dict.pkl'

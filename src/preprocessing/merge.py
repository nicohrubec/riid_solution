import pandas as pd

from src.utils import configs


def prepare_questions():
    questions = pd.read_csv(configs.questions_file,
                            dtype={
                                'question_id': 'int16',
                                'bundle_id': 'int16',
                                'correct_answer': 'int8',
                                'part': 'int8'
                            })

    questions.drop(['bundle_id', 'correct_answer'], axis=1, inplace=True)
    questions.rename(columns={'question_id': 'content_id'}, inplace=True)
    questions['content_type_id'] = 0

    return questions


def prepare_lectures():
    lectures = pd.read_csv(configs.lectures_file,
                           dtype={
                               'lecture_id': 'int16',
                               'tag': 'int16',
                               'part': 'int8'
                           })

    lectures.drop(['type_of'], axis=1, inplace=True)
    lectures.rename(columns={'lecture_id': 'content_id', 'tag': 'tags'}, inplace=True)
    lectures['content_type_id'] = 1

    return lectures


def merge_all():
    # prepare questions and lecture files for merge and put them into one df
    questions = prepare_questions()
    lectures = prepare_lectures()
    questions = questions.append(lectures)

    train = pd.read_csv(configs.train_file,
                        dtype={
                            'timestamp': 'int64',
                            'user_id': 'int32',
                            'content_id': 'int16',
                            'content_type_id': 'int8',
                            'answered_correctly': 'int8',
                            'prior_question_elapsed_time': 'float32',
                            'prior_question_had_explanation': 'boolean'
                        })

    # merge questions and lectures information so that we have all information in one big master df
    train = pd.merge(train, questions, on=['content_type_id', 'content_id'], how='left')

    # save everything to disk
    train.to_csv(configs.all_data_file, index=False)

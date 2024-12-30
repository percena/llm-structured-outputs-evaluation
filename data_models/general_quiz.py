from pydantic import BaseModel
from typing import List
from enum import Enum


class AnswerEnum(Enum):
    OPTION_0 = 0
    OPTION_1 = 1
    OPTION_2 = 2
    OPTION_3 = 3

class GeneralQuiz(BaseModel):
    question: str
    choices: List[str]
    answer: AnswerEnum

class GeneralQuizList(BaseModel):
    quizzes: List[GeneralQuiz]


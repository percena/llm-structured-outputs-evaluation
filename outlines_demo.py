from data_models.general_quiz import AnswerEnum
import outlines
import json
from tqdm import tqdm

with open("general_quizzes_20.json", "r") as file:
    quizzes = json.load(file).get("quizzes")

# print(quizzes[0])
# model = outlines.models.transformers("meta-llama/Llama-3.1-8B-Instruct")
model = outlines.models.transformers("microsoft/Phi-3-mini-128k-instruct")

correct_count = 0
incorrect_count = 0

for quiz in tqdm(quizzes):
    prompt = f"Question: {quiz['question']}\nChoices:"

    for choice in enumerate(quiz['choices']):
        prompt += f"\n{choice[0]}: {choice[1]}"

    generator = outlines.generate.choice(model, AnswerEnum)
    answer = generator(prompt)

    if answer == quiz['answer']:
        correct_count += 1
    else:
        incorrect_count += 1

print(f"Correct: {correct_count}, Incorrect: {incorrect_count}")

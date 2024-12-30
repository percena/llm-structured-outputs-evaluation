from openai import OpenAI
from dotenv import load_dotenv
from data_models.general_quiz import GeneralQuizList
from prompts.prompt_manager import PromptManager
from icecream import ic
import json


load_dotenv()
client = OpenAI()

user_message = PromptManager.get_prompt("general_quiz")

aggregated_quizzes = []

for i in range(1):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,
        response_format=GeneralQuizList,
    )
    
    quizzes = completion.choices[0].message.parsed.quizzes
    
    for quiz in quizzes:
        quiz_dict = quiz.model_dump()
        quiz_dict['answer'] = quiz.answer.value
        aggregated_quizzes.append(quiz_dict)
    
    ic(f"Run {i+1}: Retrieved {len(quizzes)} quizzes")

final_data = {
    "quizzes": aggregated_quizzes
}

with open('general_quizzes_20.json', 'w') as json_file:
    json.dump(final_data, json_file, indent=4)

ic("Aggregated quizzes have been saved to aggregated_quizzes.json")
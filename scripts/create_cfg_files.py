import os

from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

VERBOSE = True
ONE_TO_ONE_FILEPATH = "cfg/one_to_one.json"


def main(N=50):
    openai_client = OpenAI(api_key=os.getenv("DEMO_OPENAI_API_KEY"))
    chat_completion = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"List {N} words that are related to restaurants. Find words that are as descriptive as possible, and that span the entire category. Only return these {N} words, separated by a comma.",
            }
        ],
        model="gpt-3.5-turbo",
    )
    list_of_keywords = chat_completion.choices[0].message.content.split(", ")
    if VERBOSE:
        print(list_of_keywords)
    one_to_one_data = {"data": []}
    for i in range(len(list_of_keywords)):
        one_to_one_data["data"].append(list_of_keywords[i])

    # write to files
    with open(ONE_TO_ONE_FILEPATH, "w") as outputfile:
        json.dump(one_to_one_data, outputfile)


if __name__ == "__main__":
    main()

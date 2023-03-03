# NLP Assignment 02 Sean Fletcher


import re
import openai


my_key = "your key here"


def ask_chat_gpt(prompt, model_engine="davinci", max_tokens=150, temperature=0.5, n=1):
    openai.api_key = my_key
    prompt = f"{prompt.strip()}"

    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        n=n,
        temperature=temperature,
        stop=None
    )

    message = response.choices[0].text.strip()
    message = re.sub('[^0-9a-zA-Z\n\.\?,!]+', ' ', message)  # remove special characters
    return message


resp = ask_chat_gpt("What does a search warrant actually look like? How can I recognize one?")
print(resp)

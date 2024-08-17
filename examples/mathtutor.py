from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

# asst_f8rsXzkYbtrbFecrK0yAkM9E
# assistant = client.beta.assistants.create(
#     name="Math Tutor2",
#     instructions="You are a personal math tutor. Write and run code to answer math questions.",
#     tools=[{"type": "code_interpreter"}],
#     model="gpt-4o-mini",
# )
# print(assistant)

# thread_XPKX5tNFsTCeQ89TmohyrAoT
# thread = client.beta.threads.create()
# print(thread)

# msg_b0e2VJTONNO0nMTU6PvyPS2o
# message = client.beta.threads.messages.create(
#     thread_id="thread_XPKX5tNFsTCeQ89TmohyrAoT",
#     role="user",
#     content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
# )
# print(message)

# run_MOBm1L4T9pfBeeYxUHN7t7Jd
# run = client.beta.threads.runs.create(
#     thread_id="thread_XPKX5tNFsTCeQ89TmohyrAoT",
#     assistant_id="asst_f8rsXzkYbtrbFecrK0yAkM9E",
#     instructions="Please address the user as Jane Doe. The user has a premium account.",
# )
# print(run)

# run check
# run = client.beta.threads.runs.retrieve(
#     thread_id="thread_XPKX5tNFsTCeQ89TmohyrAoT",
#     run_id="run_MOBm1L4T9pfBeeYxUHN7t7Jd"
# )
# print(run)

messages = client.beta.threads.messages.list(
    thread_id="thread_XPKX5tNFsTCeQ89TmohyrAoT"
)
print(messages)

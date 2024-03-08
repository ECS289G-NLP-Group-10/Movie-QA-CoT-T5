from openai import Client
from datasets import load_dataset
import json
# Load the dataset
# dataset = load_dataset("ibm/duorc", "SelfRC")
dataset = load_dataset("ibm/duorc", "SelfRC", split='train[15:17]')
# Initialize OpenAI GPT-3.5 API client
openai_api_key = "API_KEY"
openai_client = Client(api_key=openai_api_key)

# Iterate through each instance in the dataset
for instance in dataset:
    question = instance['question']
    plot = instance['plot']
    context = f"Plot: {plot} Q: {question} A: Let’s think step by step. Please respond succinctly, using single words instead of long sentences."

    # Generate response using GPT-3.5
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": context}]
    )

    # Extract the generated answer from the response
    generated_answer = response.choices[0].message.content
    instance['answers'] = generated_answer
    print("Context", context)
    print("Question:", question)
    print("Generated Answer:", generated_answer)
    print("\n")
# WIP
# Convert the dataset to a dictionary
dataset_dict = dataset.to_dict()

# Save the dataset dictionary to a JSON file
with open('train_CoT.json', 'w') as json_file:
    json.dump(dataset_dict, json_file, indent=4)

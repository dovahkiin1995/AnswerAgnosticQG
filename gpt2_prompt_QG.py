from transformers import pipeline, set_seed
from datasets import load_dataset, load_metric
import random
from transformers import BertTokenizer, BertModel
import numpy as np
import nltk
nltk.download('stopwords') # download the stop words data
from nltk.corpus import stopwords
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--example_id",
                    default=0,
                    type=int,
                    required=False,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--kshot",
                    default=6,
                    type=int,
                    required=False,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
args = parser.parse_args()

split_type = 'validation'
squad = load_dataset('squad')
ind = args.example_id

model = BertModel.from_pretrained('bert-base-cased', output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

sentence = squad[split_type][ind]['context'] #"The capital of France is Paris. It has a population of over 2 million people and is known for its landmarks like the Eiffel Tower and the Louvre Museum."
print(sentence)
input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)

# Tokenize the sentence and convert to PyTorch tensors
#inputs = tokenizer(sentence, return_tensors='pt')

# Pass the tokenized input through the BERT model to obtain the output
outputs = model(input_ids)
attention = outputs.attentions[-1][0]  # last layer's attention scores

attention = attention.cpu().detach().numpy()
mean_attention = np.mean(attention, axis=0)
word_importance = np.sum(mean_attention, axis=0)

sorted_indices = np.argsort(word_importance)[::-1]  # sort indices in descending order
important_words = [tokenizer.convert_ids_to_tokens([input_ids[0][i].item()])[0] for i in sorted_indices[:100]]  # convert top 5 indices to words
special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=', '[', ']', '{', '}', '|', '\\', ';', ':', "'", '"', ',', '<', '.', '>', '/', '?', '[SEP]', '[CLS]']

num = 0
chosen_words = []
for word in important_words:
    if word not in special_chars and word not in stopwords.words('english') and word not in chosen_words:
        chosen_words.append(word)
        num += 1
    if num == 50:
        break

print('Top 10 chosen words:')
for word in range(0,10):
        print(chosen_words[word])

generator = pipeline('text-generation', model='gpt2-large', device=0)
sep = '[SEP]'

zero_shot_prompt_input = "The task is to generate a relevant [Contextual Question] from the given [Context] using the given [Answer Keywords].[SEP]"

one_shot_prompt_input = "The task is to generate a relevant Context Specific Question from the given [Context] using the given [Answer Keywords]. Possible answers to the generated [Contextual Question] should be relevant IN the given [Context] and MUST contain the [Answer Keywords].[SEP]" \
        "\n(1) [Context]: Many people hurt their backs when they try to lift heavy things from the floor. It is easy to hurt your back muscle when you try to pick up a heavy thing. Use [Answer Keywords]: 'back muscle' to generate [Contextual Question]: Which part of your body is more easily hurt when you lift heavy things? The [Contextual Question] is relevant to the [Context] and answer to it contains the [Answer Keywords].[SEP]"

k_shot_prompt_input = "The task is to generate a relevant Context Specific Question from the given [Context] using the given [Answer Keywords]. Possible answers to the generated [Contextual Question] should be relevant IN the given [Context] and MUST contain the [Answer Keywords].[SEP]" \
        "\n(1) [Context]: Many people hurt their backs when they try to lift heavy things from the floor. It is easy to hurt your back muscle when you try to pick up a heavy thing. Use [Answer Keywords]: 'back muscle' to generate [Contextual Question]: Which part of your body is more easily hurt when you lift heavy things? The [Contextual Question] is relevant to the [Context] and answer to it contains the [Answer Keywords].[SEP]" \
        "\n(2) [Context]: The United States Declaration of Independence is a statement adopted by the Continental Congress on July 4, 1776, which announced that the thirteen American colonies then at war with Great Britain regarded themselves as independent states, and no longer a part of the British Empire. Instead, they formed a new nation—the United States of America. Used [Answer Keywords]: 'July 4 1776' to generate [Contextual Question]: When was the United States Declaration of Independence adopted? The [Contextual Question] is relevant to the [Context] and answer to it contains the [Answer Keywords].[SEP]" \
        "\n(3) [Context]: The Taj Mahal is a white marble mausoleum in Agra, India. It was commissioned by Mughal Emperor Shah Jahan in memory of his wife Mumtaz Mahal, and completed in 1653. It is considered one of the most beautiful buildings in the world and is a symbol of love. Used [Answer Keywords]: 'Emperor Shah Jahan' to generate [Contextual Question]: Who commissioned the Taj Mahal? The [Contextual Question] is relevant to the [Context] and answer to it contains the [Answer Keywords].[SEP]" \
        "\n(4) [Context]: The Bayeux Tapestry is an embroidered cloth nearly seventy meters long, which depicts William the Conqueror's invasion of England and the events leading up to the Battle of Hastings in 1066. It is thought to have been commissioned by Bishop Odo, William's half-brother, and made in England in the 1070s. Today it is on display in the town of Bayeux in Normandy, France. Used [Answer Keywords]: 'Bishop Odo' to generate [Contextual Question]: Who might have commissioned the Bayeux Tapestry? The [Contextual Question] is relevant to the [Context] and answer to it contains the [Answer Keywords].[SEP]" \
        "\n(5) [Context]: The United Nations is an intergovernmental organization that was established in 1945. It is responsible for maintaining international peace and security, developing friendly relations among nations, and promoting social progress, better living standards and human rights. Used [Answer Keywords]: 'responsible for maintaining international peace' to generate [Contextual Question]: What is the primary responsibility of the United Nations? The [Contextual Question] is relevant to the [Context] and answer to it contains the [Answer Keywords].[SEP]" \
        "\n(6) [Context]: Soccer, also known as football, is a popular sport played around the world. Used [Answer Keywords]: 'known as football' to generate [Contextual Question]: What is soccer also known as? The [Contextual Question] is relevant to the [Context] and answer to it contains the [Answer Keywords].[SEP]"

one_shot_prompt_no_ans_input = "The task is to generate a relevant Context Specific Question from the given [Context]. Possible answers to the generated [Contextual Question] should be relevant IN the given [Context].[SEP]" \
        "\n(1) [Context]: Many people hurt their backs when they try to lift heavy things from the floor. It is easy to hurt your back muscle when you try to pick up a heavy thing. [Contextual Question]: Which part of your body is more easily hurt when you lift heavy things? The [Contextual Question] is relevant to the [Context].[SEP]"

# Change prompt type accordingly
prompt = k_shot_prompt_input
max_len = 1000
if args.kshot == 0:
    prompt = zero_shot_prompt_input
    max_len = 200
if args.kshot == 1:
    prompt = one_shot_prompt_input
    max_len = 400
q_num = "("+str(args.kshot+1)+")"
print('Original Answer Span : ' , squad[split_type][ind]['answers']['text'][0])

high_light = squad[split_type][ind]['answers']['text'][0]

# Change the number 7 according to k, also modify accordingly if no keyword is highlighted
input = prompt + "\n" + q_num + " [Context]: " + squad[split_type][ind]['context'] + " Used [Answer Keywords]: '" + high_light + "' to generate [Contextual Question]:"
result = generator(input, max_length=max_len, num_return_sequences=10)
answers = []

print('10 sample Questions generated using original span')
for i in range(10):
    answers.append(result[i]['generated_text'].split(q_num)[1].split('[Contextual Question]:')[1].split('?')[0])
    print(answers[i])

# Change the number 7 according to k, also modify accordingly if no keyword is highlighted
high_light = chosen_words[2]
print('Sample Generated Answer Span : ' , high_light)
input = prompt + "\n" + q_num + " [Context]: " + squad[split_type][ind]['context'] + " Used [Answer Keywords]: '" + high_light + "' to generate [Contextual Question]:"
result = generator(input, max_length=max_len, num_return_sequences=10)
answers = []

print('10 sample Questions generated using generated spans')
for i in range(10):
    answers.append(result[i]['generated_text'].split(q_num)[1].split('[Contextual Question]:')[1].split('?')[0])
    print(answers[i])

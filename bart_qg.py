from transformers import pipeline, set_seed
from datasets import load_dataset, load_metric
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
from transformers import pipeline
import argparse


import random
squad = load_dataset('squad')

parser = argparse.ArgumentParser()
parser.add_argument("--no_keyword",
                    default=0,
                    type=int,
                    required=False)
args = parser.parse_args()

# Print the dataset info
print(squad)

# tokenizer = AutoTokenizer.from_pretrained("p208p2002/bart-squad-qg-hl")
# Load the BART model and tokenizer
model = "p208p2002/bart-squad-qg-hl"
tokenizer_name = "p208p2002/bart-squad-qg-hl"


example_idx = 0
test_count = 256
split_type = 'validation'
high_light = '[HL]'

metric1 = load_metric("rouge")
metric2 = load_metric("bleu")
rouge_f_score = []
total_size = min(10*1024, squad[split_type].shape[0]) #squad[split_type].shape[0]

# BART no key-word
if args.no_keyword:
    test_count = 1
    eqg_generator = pipeline(
        "text2text-generation",
        model="voidful/bart-eqg-question-generator",
        tokenizer="voidful/bart-eqg-question-generator"
    )
    while (example_idx + test_count <= total_size):
        contexts = []
        target_texts = []
        source_texts = []
        for i in range(test_count):
            ind = example_idx + i
            context = squad[split_type][ind]['context']

        questions = eqg_generator(context, max_length=250, do_sample=True, num_return_sequences=10)
        print('Generated Questions : ')
        for i, question in enumerate(questions):
            print(f"Question {i + 1}: {question['generated_text'].strip()}")
        for i in range(test_count):
            ind = example_idx + i
            context = squad[split_type][ind]['context']
            question = squad[split_type][ind]['question']
            ref_list = []
            ref_list.append(question)
            target_texts.append(ref_list)
            generate_ques = questions[0]['generated_text']
            source_texts.append(generate_ques)
            if random.uniform(0, 1) < 0.01:
                print('Original Question : ', question)
                print('Generated Question : ', generate_ques)
        example_idx += test_count
        rouge_scores = metric1.compute(predictions=source_texts, references=target_texts, use_stemmer=True,
                                       use_aggregator=True)
        bleu_score = sacrebleu.corpus_bleu(source_texts, target_texts)

        print(rouge_scores['rougeL'][1])
        # print(bleu_score.score)
        rouge_f_score.append(rouge_scores['rougeL'][1][2])

    print('Rouge_score : ', sum(rouge_f_score) / len(rouge_f_score))

else:
    nlp = pipeline("text2text-generation", model=model, tokenizer=tokenizer_name, device=0)
    print('Pre-trained Model successfully loaded!')

    while(example_idx+test_count <= total_size):
        contexts = []
        target_texts = []
        source_texts = []
        for i in range(test_count):
            ind = example_idx + i
            context = squad[split_type][ind]['context']
            answer = squad[split_type][ind]['answers']['text']
            # Replace answers with the top-k obtained from keyword highlight algorithm
            new_context = context[:context.index(answer[0])] + high_light + answer[0] + high_light + ' ' + context[context.index(answer[0])+len(answer[0]):]
            contexts.append(new_context)

        questions = nlp(contexts, max_length=200, do_sample=True, num_return_sequences=3)

        for i in range(test_count):
            ind = example_idx + i
            context = squad[split_type][ind]['context']
            question = squad[split_type][ind]['question']
            ref_list = []
            ref_list.append(question)
            target_texts.append(ref_list)
            generate_ques = questions[i][0]['generated_text']
            source_texts.append(generate_ques)
            if random.uniform(0,1) < 0.01:
                print('Original Question : ', question)
                print('Generated Question : ', generate_ques)
        example_idx += test_count
        rouge_scores = metric1.compute(predictions=source_texts, references=target_texts, use_stemmer=True, use_aggregator=True)
        bleu_score = sacrebleu.corpus_bleu(source_texts, target_texts)

        print(rouge_scores['rougeL'][1])
        #print(bleu_score.score)
        rouge_f_score.append(rouge_scores['rougeL'][1][2])

    print('Rouge_score : ', sum(rouge_f_score) / len(rouge_f_score))
In order to run the code please follow the following steps

1. Clone the repo.
2. install packages: transformers, datasets, nltk, sacrebleu

The code for relevant keyword extraction is present in the file: gpt2_prompt_QG.py
The code for prompting GPT-2 for QG is also in the file: gpt2_prompt_QG.py
The prompting tempate is provided in the file 'gpt2_prompt_QG.py' with few examples. Please modify the prompt template if you want to explore further.
The code can also be ran like with the options (**--kshot** specifying 0,1,k(6) prompt type) and (**--example_id** for denoting example id)


The code for using finetuned BART: bart_qg.py
option --no_keyword is used to specify is keyword will be used for high-lighting or not. If you **don't** want to use keywords use **--no_keyword 1**

In order to run the code please follow the following steps

1. Clone the repo.
2. install packages: transformers, datasets, nltk, sacrebleu

The code for proposed keyword extraction is present in the file: gpt2_prompt_QG.py <br>
The code for prompting GPT-2 for QG is also in the file: gpt2_prompt_QG.py <br>
The prompting tempate is provided in the file 'gpt2_prompt_QG.py' with few examples. Please modify the prompt template if you want to explore further. <br>
The code can also be ran with the options (**--kshot** specifying 0,1,k(6) prompt type) and (**--example_id** for denoting example id) <br>


The code for using finetuned BART: bart_qg.py
option **--no_keyword** is used to specify if keyword will be used for high-lighting or not. If you **don't** want to use keywords (i.e. only context for question generation) use **--no_keyword 1**

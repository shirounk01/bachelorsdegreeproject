import bisect
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from word2number import w2n
import re
import wordninja

exclude_signs = [',',':',';','?']
sql_keywords = ['select', 'from', 'having', 'join', 'join left', 'join right', 'on', 'and', 'or', 'where', 'intersect', 'count', 'limit']
terminal = 'terminal'

sign_to_quantity = {
    '': '',
    '=': '',
    '!=': 'not ',
    '<>': 'not ',
    '>': 'more than/ after ',
    '<': 'less than/ before ',
    '>=': 'more or equal than ',
    '<=': 'less or equal than ',
    'like': 'similar to/ contains/ starts with/ ends with the letter/ the word/ the substirng',
    'between': 'between '
}

sql_to_question = {
    'where': '{} is {}what?/ How much?',
    'and': '{} is also {}what?',
    'or': 'What is the other {} {}?',
    'having': '{} is {}what?',
    'count(*)': '{} is {}what?',
    'limit': 'What is the limit?/ How many?'
}

model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def roberta_qna(question: str, context: str):
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    return res["answer"]

#query = input()
def terminal_to_word(query: str, context: str):
    sql_keywords_index = []
    terminal_index = []

    pairs = []
    
    for sign in exclude_signs:
        query = query.replace(sign, '')
    query_split = query.lower().split()
    #print(query_split)

    for index, word in enumerate(query_split):
        if "'"+terminal+"'" == word:
            terminal_index.append(index)
            continue
        if word in sql_keywords:
            sql_keywords_index.append(index)
    #print(terminal_index, sql_keywords_index)

    for terminal_iter in terminal_index:
        position = bisect.bisect_left(sql_keywords_index, terminal_iter)
        last_position = sql_keywords_index[position-1]
        if terminal_iter - last_position < 2 and query_split[last_position] != 'limit':
            last_position = pairs[-1][0]
        pairs.append((last_position, terminal_iter))
    #print(pairs)

    answers = []
    between_flag = False
    for pair in pairs:
        # left, right, center = pair[0], pair[1], (left+right)//2
        # print(query_split[left:right])
        # handle LIMIT
        if query_split[pair[0]] != 'limit':
            keyword_quantity = query_split[pair[0]+2]
            keyword_question = query_split[pair[0]]

            subject = ' '.join(list(filter(lambda w: len(w) > 1, wordninja.split(query_split[pair[0]+1]))))
            # subject = ' '.join(query_split[pair[0]+1].split('.'))
            quantity = sign_to_quantity[keyword_quantity]
            question = sql_to_question[keyword_question]
        else:
            question = sql_to_question['limit']

        roberta_question = question.format(subject, quantity)
        answers.append(roberta_qna(roberta_question, context))
        print(roberta_question, '\t', answers[-1])
        # # handle COUNT(*)
        # if 'count' in subject and keyword_quantity != 'between':
        #     print('oop')
        # handle LIKE
        if keyword_quantity == 'like':
            try:
                word = re.findall(r'(start|end|contain|substring|include|in it)', context)[0]
                if word == 'start':
                    answers[-1] = answers[-1] + '%'
                elif word == 'end':
                    answers[-1] = '%' + answers[-1] 
                elif word in ['contain', 'substring', 'include', 'in it']:     
                    answers[-1] = '%' + answers[-1] + '%'
            except:
                pass
        # handle BETWEEN
        if keyword_quantity == 'between':
            if between_flag == True:
                new_answer = answers[-1].split()
                for iter in range(len(new_answer)):
                    try:
                        new_answer[iter] = w2n.word_to_num(new_answer[iter])
                    except:
                        continue
                new_answer = [item for item in new_answer if isinstance(item, (int, float))]
                answers[-2], answers[-1] = new_answer[0], new_answer[1]
                
                between_flag = False
            else:
                between_flag = True

    # # handle BETWEEN and specialized cases of AND and OR
    # if len(answers) != len(set(answers)):
    #     duplicates = list(set([answer for answer in answers if answers.count(answer) > 1]))
    #     duplicates_indeces = [index for index, answer in enumerate(answers) if answer in duplicates]
    #     # duplicates_split = [[s.strip() for s in re.split(r'\b(and|or|to)\b', duplicate) if s.strip() not in ['and', 'or', 'to']] for duplicate in duplicates]
    #     # duplicates_split = [elem for sublist in duplicates_split for elem in sublist]
    #     # for duplicate, index in zip(duplicates_split, duplicates_indeces):
    #     #     answers[index] = duplicate
    #     for iter in range(len(duplicates)):
    #         duplicates[iter] = duplicates[iter].split()
    #         for iter_iter in range(len(duplicates[iter])):
    #             try:
    #                 duplicates[iter][iter_iter] = w2n.word_to_num(duplicates[iter][iter_iter])
    #             except:
    #                 continue
    #         duplicates[iter] = [item for item in duplicates[iter] if isinstance(item, (int, float))]
    #     duplicates = [elem for sublist in duplicates for elem in sublist]       
    #     for duplicate, index in zip(duplicates, duplicates_indeces):
    #         answers[index] = duplicate

    for index in range(len(answers)):
        try:
            answers[index] = w2n.word_to_num(answers[index])
        except:
            try:
                number = re.findall(r'\b\d+\b', answers[index])
                if len(number):
                    value = None
                    try:
                        value = int(number[0])
                    except:
                        value = float(number[0])
                    answers[index] = value
            except:
                continue

    for iter in answers:
        #terminal_string = terminal if not isinstance(iter, (int, float)) else "'{}'".format(terminal)
        final_answer = iter if isinstance(iter, (int, float)) else "'{}'".format(iter)
        query = query.replace("'terminal'", str(final_answer), 1)

    return query


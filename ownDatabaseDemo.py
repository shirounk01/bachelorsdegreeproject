import json
import os
import _jsonnet
import torch
from roberta2 import terminal_to_word
from ratsql.commands.infer import Inferer
from ratsql.datasets.spider import SpiderItem
from ratsql.utils import registry
import sys

db_id = sys.argv[1]
root_dir = '/content/drive/MyDrive/RatSql-Colab/rat-sql'
exp_config_path = 'experiments/spider-bert-run.jsonnet'
model_dir = '/content/drive/MyDrive/RatSql-Colab/rat-sql/logdir/bert_run/bs=2,lr=7.4e-04,bert_lr=1.0e-05,end_lr=0e0,att=1'
checkpoint_step = 81000  # whatever checkpoint you want to use
data_conf = {'db_path': '/content/drive/MyDrive/RatSql-Colab/rat-sql/data/spider/user_database',
             'name': 'spider',
             'paths': ['/content/drive/MyDrive/RatSql-Colab/rat-sql/data/spider/new_dev.json'],  # this one
             'tables_paths': ['/content/drive/MyDrive/RatSql-Colab/rat-sql/data/spider/new_tables.json']}  # and this one (should work)

exp_config = json.loads(_jsonnet.evaluate_file(exp_config_path))  # data_path: '<path to spider/>',
model_config_path = os.path.join(root_dir, exp_config["model_config"])
model_config_args = exp_config.get("model_config_args")
infer_config = json.loads(_jsonnet.evaluate_file(model_config_path, tla_codes={'args': json.dumps(model_config_args)}))

inferer = Inferer(infer_config)
inferer.device = torch.device("cpu")

model = inferer.load_model(model_dir, checkpoint_step)  # load the model according to the spider dataset
dataset = registry.construct('dataset', data_conf)  # load the new dataset (not spider)

for _, schema in dataset.schemas.items():
    model.preproc.enc_preproc._preprocess_schema(schema)

def question(q, db_id=db_id):
    schema = dataset.schemas[db_id]
    #print(schema)
    data_item = SpiderItem(
        text=None,  # intentionally None -- should be ignored when the tokenizer is set correctly
        code=None,
        schema=schema,
        orig_schema=schema.orig,
        orig={"question": q}
    )
    model.preproc.clear_items()
    enc_input = model.preproc.enc_preproc.preprocess_item(data_item, None)
    preproc_data = enc_input, None
    with torch.no_grad():
        return inferer._infer_one(model, data_item, preproc_data, beam_size=1, use_heuristic=True)

while(True):
    user_question=input('question:')
    query = question(q=user_question)[0]["inferred_code"]
    print(terminal_to_word(query=query, context=user_question))
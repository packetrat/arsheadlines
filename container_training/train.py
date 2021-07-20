import argparse
import logging
import os
import random
import sys
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def encode(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding='max_length')

# compute metrics function for binary classification
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# tokenize your inputs 
def get_token_lists(dataset):
    input_ids = []
    token_type_ids = []
    attention_masks = []
    label = []
    for i in tqdm(range(len(dataset['train']))):
        token = tokenizer.encode_plus(dataset['train']['title_a'][i],dataset['train']['title_b'][i], truncation=True, padding='max_length')
        input_ids.append(np.array(token['input_ids'], dtype=np.int32))
        token_type_ids.append(np.array(token['token_type_ids'], dtype=np.int32))
        attention_masks.append(np.array(token['attention_mask'], dtype=np.int32))
        label.append(dataset['train']['label'][i])
    return input_ids, token_type_ids, attention_masks, label

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    # Data, model, and output directories
    parser.add_argument("--checkpoints", type=str, default="/opt/ml/checkpoints/")
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=None) # os.environ["SM_CHANNEL_TEST"]
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_metric", type=str, default='accuracy')
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--fp16", type=int, default=0)

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # CHANGE ME
    try:
        dataset = load_dataset('csv', data_files=args.training_dir+'/train_df.csv')
        eval_dataset = load_dataset('csv', data_files=args.test_dir+'/eval_df.csv')
    except:
        print('had to use alternative path')
        dataset = load_dataset('csv', data_files='/opt/ml/input/data/train/train_df.csv')
        eval_dataset = load_dataset('csv', data_files='/opt/ml/input/data/test/eval_df.csv')
        
#     train_dataset = dataset.map(encode, batched=True, fn_kwargs={'tokenizer':tokenizer})
#     eval_dataset = eval_dataset.map(encode, batched=True, fn_kwargs={'tokenizer':tokenizer})
    input_ids, token_type_ids, attention_masks, label = get_token_lists(dataset)
    input_ids_e, token_type_ids_e, attention_masks_e, label_e = get_token_lists(eval_dataset)

    dataset_dict = {}
    eval_dataset_dict = {}
    dataset_dict['input_ids'] = input_ids
    dataset_dict['token_type_ids'] = token_type_ids
    dataset_dict['attention_mask'] = attention_masks
    dataset_dict['label'] = label

    eval_dataset_dict['input_ids'] = input_ids_e
    eval_dataset_dict['token_type_ids'] = token_type_ids_e
    eval_dataset_dict['attention_mask'] = attention_masks_e
    eval_dataset_dict['label'] = label_e
    
    train_dataset = Dataset.from_dict(dataset_dict)
    eval_dataset = Dataset.from_dict(eval_dataset_dict)
    columns_to_return = ['input_ids', 'label', 'attention_mask', 'token_type_ids']
    train_dataset.set_format(type='torch', columns=columns_to_return) #, device='cuda')
    eval_dataset.set_format(type='torch', columns=columns_to_return) #, device='cuda')

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}") # COMMENT OUT
    logger.info(f" loaded test_dataset length is: {len(eval_dataset)}") # COMMENT OUT

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.checkpoints,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.checkpoints}/logs",
        learning_rate=args.learning_rate,
#         load_best_model_at_end=True,
        metric_for_best_model=args.eval_metric,
        weight_decay=args.weight_decay,
        fp16=True,
    )
    
    print(train_dataset)
    print(eval_dataset)
    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=eval_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.checkpoints, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model locally. In SageMaker, writing in /opt/ml/model sends it to S3
    trainer.save_model(args.model_dir)
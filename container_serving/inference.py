import argparse
import logging
import sagemaker_containers
import requests

import os
import json
import io
import time
import glob
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# set the constants for the content types
CONTENT_TYPE = 'text/plain'


def embed_tformer(model, tokenizer, sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=256, return_tensors='pt')

    #Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

def model_fn(model_dir):
    logger.info('model_fn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_files = list(glob.glob(f"{model_dir}/*.pt"))
    logger.info(model_dir)
    logger.info(os.system(f'ls {model_dir}'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    nlp_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    nlp_model.to(device)
    model = {'model':nlp_model, 'tokenizer':tokenizer}

#     model = SentenceTransformer(model_dir + '/transformer/')
#     logger.info(model)
    return model

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(serialized_input_data, content_type=CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    try:
#         if content_type == CONTENT_TYPE:
        data = serialized_input_data.decode('utf-8') # removed list around data object
        return data
    except:
        raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    start_time = time.time()
    # need to split input object 
    print(input_object)
    inp1,inp2 = input_object.split('|')
    encoded_input = model['tokenizer'].encode_plus(text=inp1, text_pair=inp2, padding=True, truncation=True, max_length=384, return_tensors='pt')

    #Compute token embeddings
    with torch.no_grad():
        model_output = model['model'](**encoded_input)

    print("--- Inference time: %s seconds ---" % (time.time() - start_time))
    logger.info(model_output)
    response = model_output[0].tolist()
    return response

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept):
    logger.info('Serializing the generated output.')
    if accept == 'application/json':
        output = json.dumps(prediction)
        return output
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(content_type))

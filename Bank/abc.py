
import tensorflow as tf
print(tf.__version__)


# Refer: https://huggingface.co/transformers/model_doc/bert.html

from transformers import BertTokenizer, TFBertForQuestionAnswering 

modelName = 'bert-large-uncased-whole-word-masking-finetuned-squad' # https://huggingface.co/transformers/pretrained_models.html
 
tokenizer = BertTokenizer.from_pretrained(modelName)
model = TFBertForQuestionAnswering.from_pretrained(modelName)

print(model)
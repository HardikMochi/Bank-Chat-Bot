import re
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering 
def bert(ques,question):
    modelName = 'bert-large-uncased-whole-word-masking-finetuned-squad' 
    tokenizer = BertTokenizer.from_pretrained(modelName)
    bert = TFBertForQuestionAnswering.from_pretrained(modelName)

    text_for_other_question = r"""does not have minimum balance policy for most of its savings accounts. This makes it quite attractive for people to deposit their money with the bank. Capgains Plus has a minimum balance requirement of Rs. 1000/-. Savings bank account for minors does not have a minimum requirement.
    Savings bank accounts with do not attract any charges for opening or operating it. In case, there are charges applicable, the bank personnel will inform you accordingly. These policies are modified at the discretion of the bank.
    Overdraft facility is available only with select savings accounts. This feature needs to be checked before using the facility. If a cheque is drawn in excess of the account balance, it will not be paid by the bank.
    Current Account minimum balance is Rs. 5,000.The Balance Non Maintenance Charges on Current Account are Non Rural - Rs. 5,000 Rural - Rs. 2,500. The cash withdrawal limit on Current Account is .10000 per day.
    overdraft facility is available only with select savings accounts. This feature needs to be checked before using the facility. If a cheque is drawn in excess of the account balance, it will not be paid by the bank."""

    text_for_interest_rate = r"""Short-term Deposits: For an  FD with tenure ranging from 7 days to 365 days, the interest rate offered is from 4.50% p.a. to 5.80% p.a. These deposits are known as short term deposits as they have a tenure less than 1 year. For senior citizens, short term FD rates range from 5.00% p.a. to 6.30% p.a.
    Medium-term Deposits: Medium-term fixed deposits have their tenures ranging from more than 1 year to less than 5 years. The interest offered by the bank on these deposits ranges from 5.80% p.a.-6.25% p.a.
    Long-term Deposits: These deposits’ tenure ranges from 5 years to 10 years and offers an interest rate of 6.10%. Senior citizens can avail of 6.60% p.a.a for such FDs.
    Savings Deposits Balance up to Rs. 1 lakh	is Rate of Interest 2.70% p.a and Savings Deposits Balance above Rs. 1 lakh	is Rate of Interest 2.70% p.a.
    An interest rate is the percentage of principal charged by the lender for the use of its money. The principal is the amount of money loaned.
    The annual interest rate is the rate over a period of one year. Other interest rates apply over different periods, such as a month or a day, but they are usually annualised.
    A Current Account is actually a has no interest rate."""
    if ques == "interest":
        text = text_for_interest_rate
    if ques =="other":
        text = text_for_other_question
    input_text =  question + " [SEP] " + text
    input_ids = tokenizer.encode(input_text)

    print(len(input_ids))
    print(tokenizer.decode(input_ids))

    input = tf.constant(input_ids)[None, :]

    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]

    print(token_type_ids)

    answer=bert(input, token_type_ids = tf.convert_to_tensor([token_type_ids]))

    print(type(answer))
    print(len(answer))

    # https://huggingface.co/transformers/model_doc/distilbert.html#tfdistilbertforquestionanswering

    startScores = answer.start_logits
    endScores =  answer.end_logits
    print(startScores.shape)
    print(endScores.shape)
    print(type(startScores))

    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    startIdx = tf.math.argmax(startScores[0],0).numpy()
    endIdx = tf.math.argmax(endScores[0],0).numpy()+1
    print(startIdx,endIdx)
    final_answer = " ".join(input_tokens[startIdx:endIdx])
    print(final_answer) 
    return final_answer

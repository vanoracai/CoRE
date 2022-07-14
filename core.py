import joblib
import numpy as np
import scipy.special as sp
from transformers import LukeTokenizer, LukeForEntityPairClassification
import json
import torch
from tqdm import trange

# getF1 function calculates the f1 score for relation extraction.
def getF1(key, prediction):
    min_label, max_label = min(key), max(key)
    f1_ls = []
    recall_ls = []
    prec_ls = []
    for label_id_ in range(min_label, max_label + 1):
        correct_by_relation = ((key == prediction) & (prediction == label_id_)).astype(np.int32).sum()
        guessed_by_relation = (prediction == label_id_).astype(np.int32).sum()
        gold_by_relation = (key == label_id_).astype(np.int32).sum()

        prec_micro = 1.0
        if guessed_by_relation > 0:
            prec_micro = float(correct_by_relation) / float(guessed_by_relation)
        recall_micro = 1.0
        if gold_by_relation > 0:
            recall_micro = float(correct_by_relation) / float(gold_by_relation)
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)

        f1_ls.append(f1_micro)
        recall_ls.append(recall_micro)
        prec_ls.append(prec_micro)
    return sum(f1_ls[1 :]) / len(f1_ls[1 :])

# keys is the numpy array storing the sentence-wise relation labels.
# label_constraint is the numpy array storing the relation constraints based on the entity types.
# ID_TO_LABEL is the dictionary projects the label id in keys to the textual entity relations.
# These variables are stored after the pre-processing.
keys, label_constraint, ID_TO_LABEL, luke_prob_mask_2 = joblib.load('preprocess.data')

# This function loads the dataset .json file and produce the list containing the relation extraction instances.
def load_examples(dataset_file):
    with open(dataset_file, "r") as f:
        data = json.load(f)

    examples = []
    for i, item in enumerate(data):
        tokens = item["token"]
        token_spans = dict(
            subj=(item["subj_start"], item["subj_end"] + 1),
            obj=(item["obj_start"], item["obj_end"] + 1)
        )

        if token_spans["subj"][0] < token_spans["obj"][0]:
            entity_order = ("subj", "obj")
        else:
            entity_order = ("obj", "subj")

        text = ""
        cur = 0
        char_spans = {}
        for target_entity in entity_order:
            token_span = token_spans[target_entity]
            text += " ".join(tokens[cur : token_span[0]])
            if text:
                text += " "
            char_start = len(text)
            text += " ".join(tokens[token_span[0] : token_span[1]])
            char_end = len(text)
            char_spans[target_entity] = (char_start, char_end)
            text += " "
            cur = token_span[1]
        text += " ".join(tokens[cur:])
        text = text.rstrip()

        examples.append(dict(
            text=text,
            entity_spans=[tuple(char_spans["subj"]), tuple(char_spans["obj"])],
            label=item["relation"],
            entity_type = (item['subj_type'], item['obj_type']),
        ))

    return examples

# test.json is the file containing the test set of the TACRED dataset
test_examples = load_examples("test.json")

# Load the model checkpoint
model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
model.eval()
model.to("cuda")

# Load the tokenizer
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")

batch_size = 128

# produce the original testing result on the test set of TACRED
pred_ls = []
label_ls = []

for batch_start_idx in trange(0, len(test_examples), batch_size):
    batch_examples = test_examples[batch_start_idx:batch_start_idx + batch_size]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    gold_labels = [example["label"] for example in batch_examples]

    inputs = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)
    inputs = inputs.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_indices = outputs.logits.argmax(-1)
    predicted_labels = [model.config.id2label[index.item()] for index in predicted_indices]
    pred_ls.append(outputs.logits.cpu().numpy())
    label_ls = label_ls + [label_ for label_ in gold_labels]

luke_prob, luke_id_to_label = np.concatenate(pred_ls, axis = 0), model.config.id2label

# produce the counterfactual predictions to distill the entity bias
pred_ls = []
label_ls = []

for batch_start_idx in trange(0, len(test_examples), batch_size):
    batch_examples = test_examples[batch_start_idx:batch_start_idx + batch_size]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    gold_labels = [example["label"] for example in batch_examples]

    texts = [
            texts[i_][entity_spans[i_][0][0] : entity_spans[i_][0][1]] 
            + 
            ' '
            +
            texts[i_][entity_spans[i_][1][0] : entity_spans[i_][1][1]]
            for i_ in range(len(texts))
            ]
    entity_spans = [
                    [(0, entity_spans[i_][0][1] - entity_spans[i_][0][0]),
                    (entity_spans[i_][0][1] - entity_spans[i_][0][0] + 1, 
                     entity_spans[i_][0][1] - entity_spans[i_][0][0] + 1 + entity_spans[i_][1][1] - entity_spans[i_][1][0]
                    )]
                    for i_ in range(len(texts))
                   ]

    inputs = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)
    inputs = inputs.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_indices = outputs.logits.argmax(-1)
    predicted_labels = [model.config.id2label[index.item()] for index in predicted_indices]
    pred_ls.append(outputs.logits.cpu().numpy())
    label_ls = label_ls + [label_ for label_ in gold_labels]

luke_prob_mask = np.concatenate(pred_ls, axis = 0)

# normalize the predicted logits as probabilities by softmax
luke_prob = sp.softmax(luke_prob, axis = 1)
luke_prob_mask = sp.softmax(luke_prob_mask, axis = 1)

# transform the luke prediction indices to the original label indices.
luke_label_to_id = {value_ : key_ for key_, value_ in luke_id_to_label.items()}
org_to_luke = [luke_label_to_id[ID_TO_LABEL[i_]] for i_ in range(len(luke_label_to_id.values()))]
luke_prob = luke_prob[:, org_to_luke]
luke_prob_mask_1 = luke_prob_mask[:, org_to_luke]
luke_preds = luke_prob.argmax(1)
luke_preds_tde = luke_prob_mask_1.argmax(1)

# filter the challenge set that the relation labels implied by the entity bias does not exist in the sentence.
challenge_set = [i_ for i_ in range(len(luke_prob)) if luke_preds_tde[i_] != keys[i_]]

keys = keys[challenge_set]
luke_preds = luke_preds[challenge_set]
luke_prob = luke_prob[challenge_set]
luke_prob_mask_1 = luke_prob_mask_1[challenge_set]
luke_prob_mask_2 = luke_prob_mask_2[challenge_set]
label_constraint = label_constraint[challenge_set]

print('f1 score before bias mitigation: ', getF1(keys, luke_preds))

lamb_1 = -1.6
lamb_2 = 0.1
new_preds = (luke_prob + lamb_1 * luke_prob_mask_1 + lamb_2 * luke_prob_mask_2 + label_constraint).argmax(1)
print('f1 score after bias mitigation: ', getF1(keys, new_preds))
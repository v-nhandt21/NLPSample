from utils import LoadData
from utils import load_checkpoint
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from torchcrf import CRF
from model import AttentionBiLSTM_Seqence
import torchtext
from seq_scorer import score

train_iter, valid_iter, test_iter, TAGS, TEXT, fields = LoadData()

device = "cuda"
count = 0
    
def check_infer():
    with open("Sequence_Labeling/data/test.label", "r", encoding="utf-8") as fr:
        text_infer = fr.read().splitlines()
    with open("Sequence_Labeling/data/predict.txt", "r", encoding="utf-8") as fr1:
        text_infer1 = fr1.read().splitlines()
    for t, t1 in zip(text_infer, text_infer1):
        
        assert len(t.split(" ") ) - 1 == len(t1.split(" "))

def evaluate(model):
        # This method applies the trained model to a list of sentences.
        
        # First, create a torchtext Dataset containing the sentences to tag.
        crf = CRF(len(TAGS.vocab)).to(device)

        model.eval()
        out = []
        with open("Sequence_Labeling/data/predict_bilstm.txt", "w+", encoding="utf-8") as fw:
            with torch.no_grad():
                for (text,tags), _ in test_iter:

                    output = model(text,tags)
                    top_predictions = crf.decode(output)

                    predicted_tags = [TAGS.vocab.itos[t] for t in top_predictions[0] ]

                    fw.write(" ".join(predicted_tags) + "\n")


if __name__ == '__main__':

    best_model = AttentionBiLSTM_Seqence(TEXT, TAGS, n_embed=300, n_hidden=128).to("cuda")

    load_checkpoint('Seq_BiLSTM.pt', best_model)

    evaluate(best_model)

    check_infer()

    key = [str(line).rstrip('\n') for line in open("Sequence_Labeling/data/test.label")]
    prediction = [str(line).rstrip('\n') for line in open("Sequence_Labeling/data/predict.txt")]
    score(key, prediction, verbose=True)
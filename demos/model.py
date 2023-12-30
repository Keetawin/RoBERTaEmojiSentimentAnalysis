
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class SentimentAnalysisModel(torch.nn.Module):
    def __init__(self, bert_model_name, num_labels=7, dropout_rate=0.4):
        super(SentimentAnalysisModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(bert_model_name)
        self.model.classifier.out_proj = torch.nn.Linear(self.model.config.hidden_size, num_labels)
        self.model.classifier.dropout = torch.nn.Dropout(p=dropout_rate, inplace=False)
        # ปรับ dropout ในทุก RobertaLayer
        for layer in self.model.roberta.encoder.layer:
            layer.attention.self.dropout = torch.nn.Dropout(p=dropout_rate)
            layer.attention.output.dropout = torch.nn.Dropout(p=dropout_rate)
            layer.intermediate.dropout = torch.nn.Dropout(p=dropout_rate)
            layer.output.dropout = torch.nn.Dropout(p=dropout_rate)

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

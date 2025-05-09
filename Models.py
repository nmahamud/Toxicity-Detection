from torch import nn
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import Constants

class distilRB_base(nn.Module):
    def __init__(self, multilabel: bool=False):
        self.base_model = AutoModel.from_pretrained("distilbert/distilroberta-base")
        self.activation = nn.Linear(self.base_model.config.hidden_size, Constants.NUM_CLASSES if multilabel else 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        inter = self.base_model(input_ids, attention_mask)
        pooled = inter.pooler_output
        y_hat = self.activation(pooled)
        return self.sigmoid(y_hat)

class distilRB_hate(nn.Module):
    def __init__(self, multilabel: bool=False):
        self.base_model = AutoModel.from_pretrained("distilbert/distilroberta-base")
        self.hate_model = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_roberta") 
        #self.hate_model output for forward is a logit of size 2
        self.activation = nn.Linear(self.base_model.config.hidden_size + Constants.NUM_HATE, Constants.NUM_CLASSES if multilabel else 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        inter = self.base_model(input_ids, attention_mask)
        pooled = inter.pooler_output
        hate_signal = self.hate_model(input_ids, attention_mask)
        combined = torch.cat([pooled, hate_signal], dim=1)

        y_hat = self.activation(combined)
        return self.sigmoid(y_hat)


class distilRB_sem(nn.Module):
    def __init__(self, multilabel: bool=False):
        self.base_model = AutoModel.from_pretrained("distilbert/distilroberta-base")
        self.sent_anal = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        #self.sentiment output for forward is a logit of size 3
        self.activation = nn.Linear(self.base_model.config.hidden_size + Constants.NUM_SENTIMENT, Constants.NUM_CLASSES if multilabel else 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        inter = self.base_model(input_ids, attention_mask)
        pooled = inter.pooler_output
        sentiment = self.sent_anal(input_ids, attention_mask)
        combined = torch.cat([pooled, sentiment], dim=1)
        y_hat = self.activation(combined)
        return self.sigmoid(y_hat)        

class distilRB_combine(nn.Module):
    def __init__(self, multilabel: bool=False):
        self.base_model = AutoModel.from_pretrained("distilbert/distilroberta-base")
        self.sent_anal = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.hate_model = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_roberta") 
        #self.sentiment output for forward is a logit of size 3
        self.activation = nn.Linear(
            self.base_model.config.hidden_size + Constants.NUM_SENTIMENT + Constants.NUM_HATE, Constants.NUM_CLASSES if multilabel else 1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        inter = self.base_model(input_ids, attention_mask)
        pooled = inter.pooler_output
        sentiment = self.sent_anal(input_ids, attention_mask)
        hate = self.hate_model(input_ids, attention_mask)
        combined = torch.cat([pooled, sentiment, hate], dim=1)
        y_hat = self.activation(combined)
        return self.sigmoid(y_hat)  

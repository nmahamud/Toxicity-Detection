from torch import nn
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

NUM_CLASSES = 6
NUM_SENTIMENT = 3
NUM_HATE = 2

BATCH_SIZE=32
EPOCHS=1

class distilRB_base(nn.Module):
    def __init__(self, multilabel: bool=False):
        super(distilRB_base, self).__init__()
        self.base_model = AutoModel.from_pretrained("distilbert/distilroberta-base")
        self.activation = nn.Linear(self.base_model.config.hidden_size, NUM_CLASSES if multilabel else 1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        inter = self.base_model(input_ids, attention_mask)
        pooled = inter.pooler_output
        y_hat = self.activation(pooled)
        return y_hat

class distilRB_hate(nn.Module):
    def __init__(self, multilabel: bool=False):
        super(distilRB_hate, self).__init__()
        self.base_model = AutoModel.from_pretrained("distilbert/distilroberta-base")
        self.hate_model = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_roberta") 
        for param in self.hate_model.parameters():
            param.requires_grad = False
        #self.hate_model output for forward is a logit of size 2
        self.activation = nn.Linear(self.base_model.config.hidden_size + NUM_HATE, NUM_CLASSES if multilabel else 1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        inter = self.base_model(input_ids, attention_mask)
        pooled = inter.pooler_output
        hate_signal = self.hate_model(input_ids, attention_mask)
        combined = torch.cat([pooled, hate_signal], dim=1)

        y_hat = self.activation(combined)
        return y_hat


class distilRB_sem(nn.Module):
    def __init__(self, multilabel: bool=False):
        super(distilRB_sem, self).__init__()
        self.base_model = AutoModel.from_pretrained("distilbert/distilroberta-base")
        self.sent_anal = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        for param in self.sent_anal.parameters():
            param.requires_grad = False
        #self.sentiment output for forward is a logit of size 3
        self.activation = nn.Linear(self.base_model.config.hidden_size + NUM_SENTIMENT, NUM_CLASSES if multilabel else 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        inter = self.base_model(input_ids, attention_mask)
        pooled = inter.pooler_output
        sentiment = self.sent_anal(input_ids, attention_mask)
        combined = torch.cat([pooled, sentiment], dim=1)
        y_hat = self.activation(combined)
        return y_hat

class distilRB_combine(nn.Module):
    def __init__(self, multilabel: bool=False):
        super(distilRB_combine, self).__init__()
        self.base_model = AutoModel.from_pretrained("distilbert/distilroberta-base")
        self.sent_anal = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.hate_model = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_roberta") 
        for param in self.sent_anal.parameters():
            param.requires_grad = False

        for param in self.hate_model.parameters():
            param.requires_grad = False
        #self.sentiment output for forward is a logit of size 3
        self.activation = nn.Linear(
            self.base_model.config.hidden_size + NUM_SENTIMENT + NUM_HATE, NUM_CLASSES if multilabel else 1
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        inter = self.base_model(input_ids, attention_mask)
        pooled = inter.pooler_output
        sentiment = self.sent_anal(input_ids, attention_mask)
        hate = self.hate_model(input_ids, attention_mask)
        combined = torch.cat([pooled, sentiment, hate], dim=1)
        y_hat = self.activation(combined)
        return y_hat

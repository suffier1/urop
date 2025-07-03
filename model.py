import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2Model

class AudioTextEmotionModel(nn.Module):
    def __init__(self, text_model_name='distilbert-base-uncased', audio_model_name='facebook/wav2vec2-base-960h', num_labels=4):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.audio_model = Wav2Vec2Model.from_pretrained(audio_model_name)
        hidden_size = self.text_model.config.hidden_size + self.audio_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, input_values, labels=None):
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
        audio_out = self.audio_model(input_values=input_values).last_hidden_state[:,0]
        hidden = torch.cat([text_out, audio_out], dim=1)
        logits = self.classifier(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {'loss': loss, 'logits': logits}

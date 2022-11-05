from sklearn.semi_supervised import LabelSpreading
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIP(nn.Module):

    def __init__(self, image_encoder, text_encoder, temperature) -> None:
        super(CLIP, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_projection = nn.Linear(512, 512)
        self.text_projection = nn.Linear(512, 512)
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, image, text): 
        image = self.image_encoder(image)
        text = self.text_encoder(text)
        image = self.image_projection(image)
        text = self.text_projection(text)
        image = F.normalize(image, dim=-1)
        text = F.normalize(text, dim=-1)
        logits = (image @ text.T) * torch.exp(self.temperature)
        labels = torch.arange(logits.shape[0]).to(logits.device)
        loss_i = self.ce_loss(logits, labels)
        loss_t = self.ce_loss(logits.T, labels)
        return (loss_i + loss_t) / 2.0
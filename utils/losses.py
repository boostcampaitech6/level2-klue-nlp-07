import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Trainer

#focal loss code : https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLossTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    # focal loss
    # 30 labels
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits

    alpha = 0.25
    gamma = 2.0
    
    ce_loss = torch.nn.CrossEntropyLoss()
    ce_loss = ce_loss(logits.view(-1, self.model.config.num_labels), labels.view(-1))

    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss

    return (focal_loss, outputs) if return_outputs else focal_loss      
  

# LabelSmoothing Loss  
class LabelSmoothingLossTrainer(Trainer):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLossTrainer, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        pred = logits.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, labels.data.unsqueeze(1), self.confidence)

        smoothing_loss = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

        return (smoothing_loss, outputs) if return_outputs else smoothing_loss
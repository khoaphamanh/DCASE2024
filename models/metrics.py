import torch
from torch import nn
import torch.nn.functional as F


# Arcface Loss
class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, emb_size, margin, scale, class_weights=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.margin = margin
        self.scale = scale
        self.w = nn.Parameter(
            data=torch.randn(size=(num_classes, emb_size)), requires_grad=True
        ).to(self.device)
        self.class_weights = class_weights.to(self.device)

    def forward(self, embedding, y_true):

        # calculate logits
        logits = self.logits(embedding=embedding, y_true=y_true)

        # combine with cross entropy loss
        ce = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = ce(logits, y_true)

        return loss

    def logits(self, embedding, y_true):

        # cos(phi) =  (x @ w.t) / (||w.t||.||x|| ) = normalize(x) @ normalize(w.t) / 1 beacause (||normalize(w.T)|| = ||normalize(x)|| )
        cosine_logits = F.linear(
            input=F.normalize(embedding), weight=F.normalize(self.w)
        )

        # onehot vector based on y_true
        onehot = self.onehot_true_label(y_true)  # size (B, num_classes)

        # cosine logit of the target class index
        cosine_target = cosine_logits[onehot == 1]  # size (B,)

        # calculate cosine phi in target class index with phi = angle + m
        cosine_phi = self.cosine_angle_plus_margin(
            cosine_target=cosine_target
        )  # size (B,)

        # calculate logit new
        diff = (cosine_phi - cosine_target).unsqueeze(1)
        logits = cosine_logits + (onehot * diff)  # size (B,num_classes)
        logits = self.scale * logits

        return logits

    def onehot_true_label(self, y_true):
        batch_size = y_true.shape[0]
        onehot = torch.zeros(batch_size, self.num_classes).to(self.device)
        onehot.scatter_(1, y_true.unsqueeze(-1), 1)
        return onehot

    def cosine_angle_plus_margin(self, cosine_target):
        eps = 1e-7
        angle = torch.acos(torch.clamp(cosine_target, -1 + eps, 1 - eps))
        phi = angle + self.margin
        cosine_phi = torch.cos(phi)
        return cosine_phi


# Adacos Loss
class AdaCosLoss(nn.Module):
    def __init__(self, num_classes, emb_size, class_weights=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.w = nn.Parameter(
            data=torch.randn(size=(num_classes, emb_size)), requires_grad=True
        ).to(self.device)
        self.class_weights = class_weights.to(self.device)
        self.scale = torch.sqrt(torch.tensor(2.0)) * torch.log(
            torch.tensor(num_classes - 1)
        )

    def forward(self, embedding, y_true):

        # logits
        cosine_logits = self.logits(embedding)  # size (B, n_classes)

        # angle from cosine_logits
        angle = self.angle(cosine_logits)

        # onehot vector based on y_true
        onehot = self.onehot_true_label(y_true)  # size (B, n_classes)

        # new scale
        if self.training:
            with torch.no_grad():
                # B_avg
                batch_size = y_true.shape[0]
                B_avg = torch.where(
                    onehot < 1,
                    torch.exp(self.scale * cosine_logits),
                    torch.zeros_like(cosine_logits),
                )  # size (B, n_classes)
                B_avg = torch.sum(B_avg) / batch_size  # size (1,)

                # medium of the angles of true labels
                angle_median = torch.median(angle[onehot == 1])  # size (1,)

                # update scale
                self.scale = torch.log(B_avg) / torch.cos(
                    torch.min(
                        torch.pi / 10 * torch.ones_like(angle_median),
                        angle_median,
                    )
                )

        # calculate new logits
        logits = self.scale * cosine_logits

        # apply cross entropy loss
        ce = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = ce(logits, y_true)

        return loss

    def logits(self, embedding, y_true=None):
        # cos(phi) =  (x @ w.t) / (||w.t||.||x|| ) = normalize(x) @ normalize(w.t) / 1 beacause (||normalize(w.T)|| = ||normalize(x)|| )
        cosine_logits = F.linear(
            input=F.normalize(embedding), weight=F.normalize(self.w)
        )  # size (B, n_classes)
        return cosine_logits

    def angle(self, cosine_logits):
        # angle from given cosine logits
        eps = 1e-7
        angle = torch.acos(torch.clamp(cosine_logits, -1 + eps, 1 - eps))
        return angle

    def onehot_true_label(self, y_true):
        """
        y_true = [0,2,1]
        n_classes = 10
        onehot = [[1,0,0,0,0,0,0,0,0,0],
                  [0,0,1,0,0,0,0,0,0,0],
                  [0,1,0,0,0,0,0,0,0,0]]
        """
        batch_size = y_true.shape[0]
        onehot = torch.zeros(batch_size, self.num_classes).to(self.device)
        onehot.scatter_(1, y_true.unsqueeze(-1), 1)
        return onehot

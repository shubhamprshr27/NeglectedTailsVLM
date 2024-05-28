import torch
from torch import nn

class MyLinear(nn.Module):
    def __init__(self, weights=None, inp_dim=512, num_classes=810, bias = True, label_map = None):
        super(MyLinear, self).__init__()
        
        if torch.is_tensor(weights):
            self.linear = nn.Linear(weights.shape[1], weights.shape[0], bias=bias) # Set bias = False, so that we simply do Zero Shot.
            with torch.no_grad():
                self.linear.weight.copy_(weights)
            self.num_classes = weights.shape[0]
        else:
            self.linear = nn.Linear(inp_dim, num_classes, bias=bias)
            self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=-1)
        self.label_map = label_map
        if self.label_map is not None:
            self.all_indices = torch.cat([torch.tensor(label_map[label]) for label in self.label_map], dim=0)
            self.segment_lengths = torch.tensor([len(indices) for indices in self.label_map.values()])
        # self.logit_scale = torch.FloatTensor([4.6017]).cuda()
    
    def forward(self, x):
        x = self.linear(x)
        if self.label_map is not None:
            segments = torch.split_with_sizes(x, self.segment_lengths.tolist(), dim=1)

            # Compute max for each segment and collect in a list
            max_list = [segment.max(dim=1).values for segment in segments]

            segmented_maxes = torch.stack(max_list)
            x = segmented_maxes.t()
        # x = self.softmax(x)
        # x = x * self.logit_scale.exp()
        return x
    
    def update_weights(self, weights): # Cosine Similarity Validation during CLIP fine-tuning. 
        with torch.no_grad():
            self.linear.weight.copy_(weights)
    


    
         
         
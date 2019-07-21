import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        
        # LSTM cell
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers=self.num_layers)
        
        # output FC 
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
    
    def forward(self, features, captions):
        # batch size
        batch_size = features.size(0)
        
        # init the hidden and cell states to zeros (num_layers, batch_size, hidden_dim)
        hidden_state = torch.zeros((self.num_layers, batch_size, self.hidden_size))
        cell_state = torch.zeros((self.num_layers, batch_size, self.hidden_size))
    
        # define the output tensor placeholder
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size))
        
        # Embedding the input captions
        captions = self.emb(captions)
        
        # pass the caption word by word
        for t in range(captions.size(1)):
            # First pass the CNN's output
            if t == 0:
                hidden_state, cell_state = self.lstm(features.view(1, batch_size, -1), (hidden_state, cell_state))
                out = self.fc_out(hidden_state)
            # Then pass word's caption
            else:
                hidden_state, cell_state = self.lstm(captions[:, t-1, :].view(1, batch_size, -1), cell_state)
                out = self.fc_out(hidden_state)
                
            outputs[:, t, :] = out
    
        return outputs   

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # init the hidden and cell states to zeros (num_layers, batch_size, hidden_dim)
        hidden_state = torch.zeros((self.num_layers, 1, self.hidden_size))
        cell_state = torch.zeros((self.num_layers, 1, self.hidden_size))
        
        with torch.no_grad():
            i = 0
            predictions = []

            while i < max_len:
                # Pass image features
                if i == 0:
                    hidden_state, cell_state = self.lstm(inputs.view(1, 1, -1), (hidden_state, cell_state))
                    pred = self.fc_out(hidden_state)
                    pred_int = torch.argmax(pred, dim=2)
                    predictions.append(int(pred_int))
                else:
                    pred_emb = self.emb(torch.Tensor([[int(pred_int)]]).long().to('cpu'))
                    hidden_state, cell_state = self.lstm(pred_emb.view(1, 1, -1), cell_state)
                    pred = self.fc_out(hidden_state)
                    pred_int = torch.argmax(pred, dim=2)
                    predictions.append(int(pred_int))
                # Stop prediction if end token is predicted
                if int(pred_int) == 1:
                    break
                i += 1
                
        return predictions
        
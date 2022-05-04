import torch.nn as nn

#------------------------------
#   The Generator as in 
#   https://www.aclweb.org/anthology/2020.acl-main.191/
#   https://github.com/crux82/ganbert
#------------------------------
class Generator(nn.Module):
    def __init__(self, noise_size=100, output_size=512, hidden_sizes=[512], dropout_rate=0.1):
        super(Generator, self).__init__()
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        # the generator is an MLP with 100 hidden ---> 512 hidden (could be any number of layers) ---> leaky relu activation ---> dropout
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])
        
        # the last hidden layer is then created from the output size of the previous layer (512) followed by the output size (512)
        layers.append(nn.Linear(hidden_sizes[-1],output_size))
        
        # final MLP in this case is 100 X 512 X 512
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        # Input is 100 dim and output is 512 dim
        output_rep = self.layers(noise)
        return output_rep

#------------------------------
#   The Discriminator
#   https://www.aclweb.org/anthology/2020.acl-main.191/
#   https://github.com/crux82/ganbert
#------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[512], num_labels=2, dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        
        # the input size is 512
        hidden_sizes = [input_size] + hidden_sizes
        
        for i in range(len(hidden_sizes)-1):
            print(f"hidden_sizes[i] and [i+1] is {hidden_sizes[i]} and {hidden_sizes[i+1]}")
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        # input 512 X 512
        self.layers = nn.Sequential(*layers) #per il flatten
        # the concluding layer will have 512 X 4 (for labels 0, 1, UNK, real/fake)
        self.logit = nn.Linear(hidden_sizes[-1],num_labels+1) # +1 for the probability of this sample being fake/real.
        self.softmax = nn.Softmax(dim=-1)
        
        # final discriminator will have dropout layer ---> 512 X 512 X 4 ---> fed to softmax ----> actual probs for the labels

    def forward(self, input_rep):
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs
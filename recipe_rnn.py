#Learn to model recipes and generate new ones via lstm

import os, sys, re
import time, math, random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import argparse

import numpy as np

#Recipe RNN utility functions

#standard line cleaning
def clean_line(line):
    return line.replace("'","").replace(",","").replace(":","").replace(";","").replace("(","").replace(")","").replace("&#39", "").replace("\n", " <EOL>").lower()

#Clean and concatenate recipe files
def concat_recipes(recipe_files, outfile_name):
    allrecipes_file = open(outfile_name, "w")
    for recipe in recipe_files:
        #print(recipe)
        for line in open(recipe):
            line = clean_line(line)
            allrecipes_file.write(line + " ")

        allrecipes_file.write("<eos> ")
    allrecipes_file.close()


#Argument parsing from pytorch language modeling example code
parser = argparse.ArgumentParser(description='Recipe RNN')
parser.add_argument('--data', type=str, default='./allrecipes_downloads/',
                    help='location of the data corpus')
parser.add_argument('--embedding', type=str, default='./glove_recipe_vectors_100.txt',
                    help='Path of GloVe embedding definition file')
parser.add_argument('--emsize', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--iters', type=int, default=20000,
                    help='number of iterations to trail')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--gpu', action='store_true', default = True,
                    help='use GPU')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--load', type=str, default=None,
                    help='path to load existing model from')
parser.add_argument('--prime_str', type=str, default="chocolate",
                    help='String to prime recipe generation with')
parser.add_argument('--print_every', type=int, default=100,
                    help='Frequency to test/output stats during training')
parser.add_argument('--concat', type=str, default=None,
                    help='Clean and concatenate recipe files for embedding processing, then exit')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

gpu = 0

recipe_dir = args.data

recipe_files = [os.path.join(recipe_dir, name) for name in os.listdir(recipe_dir) if "_ingredients.txt" in name]

if(args.concat is not None):
    concat_recipes(recipe_files, args.concat)
    print("Recipe files concatenated, now generate an embedding with GloVe")
    exit()


#Build dictionary mapping words to indexes/ID numbers
word_dict = dict()
#Mapping from indices to words
ind_2_word = dict()


print("Loading embedding...")

#Load glove embedding from file
glove_filename = args.embedding

#get number of lines
num_lines = sum(1 for line in open(glove_filename))
#Could read this from file...
embedding_dim = args.emsize

embedding_tens = torch.zeros(num_lines, embedding_dim)

glove_file = open(glove_filename)
line = glove_file.readline()
for line_n in range(num_lines):
    line = line.replace("\n","").split(" ")
    #first token is word
    word = line[0]
    word_dict[word] = line_n
    ind_2_word[line_n] = word
    
    embedding_tens[line_n] = torch.Tensor([float(i) for i in line[1:]])

    line = glove_file.readline()

#Finally we have an embedding layer!    
#embedding = nn.Embedding.from_pretrained(embedding_tens, freeze = True)
embedding = nn.Embedding(embedding_tens.size(0),embedding_tens.size(1))
embedding.weight = nn.Parameter(embedding_tens)
#embedding.weight.requires_grad = False
embedding = embedding

print("Embedding loaded.")

#Define RNN model
class recipeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, embedding, tie_weights = False, drop_freq = 0.5):

        super(recipeRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        #Make the decoder linear operation be the inverse of the embedding
        #Requires the rnn to have hidden size equal to the embedding length
        self.tie_weights = tie_weights

        self.drop = nn.Dropout(drop_freq)

        self.encoder = embedding
        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first = True, dropout = drop_freq)
          
        self.decoder = nn.Linear(hidden_size, output_size)
        if(self.tie_weights):
            self.decoder.weight = self.encoder.weight


    def forward(self, input, hidden):
        encoded = self.encoder(input)
        encoded = self.drop(encoded)
        if(len(encoded.size()) < 3):
            encoded = encoded.unsqueeze(0)
        #print(encoded, hidden)
        output, hidden = self.rnn(encoded, hidden)
        output = self.drop(output)
        output = self.decoder(output.squeeze())
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                torch.zeros(self.n_layers, batch_size, self.hidden_size))


#Let's load and clean our recipe dataset into a list of tensors
if(args.load is None):
    print("Loading recipes...")
    recipe_tensors = []
    for recipe in recipe_files:
        #print(recipe)
        tokens = []
        for line in open(recipe):
            line = clean_line(line).split(" ")
            for tok in line:
                if(tok in word_dict.keys()):
                    tokens.append(word_dict[tok])
                else: #unknown word
                    tokens.append(word_dict["<unk>"])
        tokens.append(word_dict["<eos>"])
        var = torch.LongTensor(tokens)
        if(args.gpu):
            var = var.cuda(gpu)
        l = var.size(0)
        recipe_tensors.append((l + random.random() * 0.0001, var)) #include lengths to sort by

    #Sort and extract tensors only
    #print([ele[0] for ele in sorted(recipe_tensors)])
    recipe_tensors = [ele[1] for ele in sorted(recipe_tensors, reverse = True)]
    recipe_tensors = torch.nn.utils.rnn.pad_sequence(recipe_tensors).permute(1,0)
    print("Loaded recipes")

#Set up and train!



#n_epochs = 20000
#chunk_len = 200
#print_every = 100
#batch_size = 100

#clip_cap = 0.25
#hidden_size = 100
#learning_rate = 0.01
#n_layers = 3
recipe_model =  recipeRNN(
    args.emsize,
    args.nhid,
    len(word_dict.keys()),
    n_layers = args.nlayers,
    embedding = embedding,
    tie_weights = args.tied,
    drop_freq = args.dropout
)
if(args.gpu):
    recipe_model = recipe_model.cuda(gpu)
opt = torch.optim.Adam(recipe_model.parameters(), lr=args.lr)
criterion = F.cross_entropy


#pre-allocated Tensor to hold current batch
if(args.load is None):
    curr_batch = torch.LongTensor(args.batch_size, recipe_tensors.size(1))
    if(args.gpu):
        curr_batch = curr_batch.cuda(gpu)

#Train by predicting next word given current word and hidden state
#Batches currently handled by running on multiple sequences in series, accumulating gradient updates before calling opt. 
#May not be the most efficient solution...
def train(inputs):
    hidden, cell = recipe_model.init_hidden(args.batch_size)
    if(args.gpu):
        hidden = hidden.cuda(gpu)
        cell = cell.cuda(gpu)
    hidden = (hidden, cell)
    recipe_model.zero_grad()
    loss = 0

    for b in range(args.batch_size):
        rand_ind = int(math.floor(random.random() * len(inputs)))
        curr_seq = inputs[rand_ind]
        curr_batch[b] = curr_seq
        
    ind = 0
    loss = 0
    while ind + 1 < curr_batch.size(1):

        output, hidden = recipe_model(curr_batch[:,ind:ind+1], hidden)
        loss += criterion(output, curr_batch[:,ind+1])
        ind += 1
    #targets = Variable(curr_seq.data.new().resize_(curr_seq.size()))
    #targets[0,0:targets.size(1) - 1] = curr_seq[0,1:]
    #targets[0,targets.size(1) - 1] = word_dict["<eol>"]

    #output, hidden = recipe_model(curr_seq, hidden)
    #loss += criterion(output.squeeze(), targets.squeeze())
    loss.backward()

    #Clip gradients in case of exploding gradients
    torch.nn.utils.clip_grad_norm(recipe_model.parameters(), args.clip)

    #hidden, cell = recipe_model.init_hidden(1)
    #if(args.gpu):
    #    hidden = hidden.cuda(gpu)
    #    cell = cell.cuda(gpu)
    #hidden = (hidden, cell)

    opt.step()

    return loss.data.item()

#===Generate Text

def generate(recipe_model, prime_str='a', predict_len=60, temperature=0.8, cuda=False):

    #parse the priming string
    tokens = []
    prime_tok = clean_line(prime_str).split(" ")
    for tok in prime_tok:
        if(tok in word_dict.keys()):
            tokens.append(word_dict[tok])
        else: #unknown word
            tokens.append(word_dict["<unk>"])
    prime_input = torch.LongTensor(tokens).unsqueeze(0)

    hidden, cell = recipe_model.init_hidden(1)
    if cuda:
        hidden = hidden.cuda(gpu)
        cell = cell.cuda(gpu)
        prime_input = prime_input.cuda()
    hidden = (hidden, cell)
    predicted = prime_str

    for p in range(len(prime_tok) - 1):
        _, hidden = recipe_model(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = recipe_model(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = ind_2_word[top_i.item()]
        if(predicted_char == "<eos>"):
            break
        predicted += " " + predicted_char.replace("<eol>","\n")
        inp = torch.LongTensor([word_dict[predicted_char]]).unsqueeze(0)
        if cuda:
            inp = inp.cuda()

    return predicted


#===Train and run!        

print(recipe_model)

start = time.time()
all_losses = []
loss_avg = 0

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if(args.load is not None):
    print("Loading weights from file...")
    weights_filename = args.load
    recipe_model.load_state_dict(torch.load(weights_filename))
else:

    print("Training for %d epochs..." % args.iters)
    for epoch in range(1, args.iters + 1):
        recipe_model.train(True)
        loss = train(recipe_tensors)
        loss_avg += loss

        if epoch % args.print_every == 0:
            recipe_model.train(False)
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.iters * 100, loss_avg / args.print_every))
            print('loss: ', loss)
            print(generate(recipe_model, args.prime_str, 40, 0.8, cuda=args.gpu) + "\n")
            loss_avg = 0

    #Save model to file
    torch.save(recipe_model, args.save + ".model")
    torch.save(recipe_model.state_dict(), args.save + ".weights")

#"""### Let's try sampling with high temperature:"""

#print(generate(recipe_model, prime_str="a", temperature= 0.8, cuda=args.gpu))

#"""### Let's try sampling with low temperature:"""
recipe_model.train(False)
for prop in range(10):
    print(generate(recipe_model, prime_str=args.prime_str, temperature= 0.8, cuda=args.gpu))


#print(generate(recipe_model, prime_str="sugar", cuda=args.gpu))

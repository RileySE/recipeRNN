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

#Load glove embedding from file
def load_glove_embedding(glove_filename, embedding_dim):

    #get number of lines
    num_lines = sum(1 for line in open(glove_filename))

    embedding_tens = torch.zeros(num_lines, embedding_dim)

    word_2_ind = dict()
    ind_2_word = dict()

    glove_file = open(glove_filename)
    line = glove_file.readline()
    for line_n in range(num_lines):
        line = line.replace("\n","").split(" ")
        #first token is word
        word = line[0]
        word_2_ind[word] = line_n
        ind_2_word[line_n] = word
    
        embedding_tens[line_n] = torch.Tensor([float(i) for i in line[1:]])

        line = glove_file.readline()

    #Finally we have an embedding layer!    
    embedding = nn.Embedding(embedding_tens.size(0),embedding_tens.size(1))
    embedding.weight = nn.Parameter(embedding_tens)
    #embedding.weight.requires_grad = False

    return embedding, word_2_ind, ind_2_word

#Clean and load recipe/text files
def load_recipes(recipe_files, word_2_ind, max_length = 100, gpu = 0):

    recipe_tensors = []
    for recipe in recipe_files:
        #print(recipe)
        tokens = []
        for line in open(recipe):
            line = clean_line(line).split(" ")
            for tok in line:
                if(tok in word_2_ind.keys()):
                    tokens.append(word_2_ind[tok])
                else: #unknown word
                    tokens.append(word_2_ind["<unk>"])
        tokens.append(word_2_ind["<eos>"])
        if(len(tokens) == 0):
            continue
        elif(len(tokens) > max_length):
            continue
        var = torch.LongTensor(tokens)
        if(gpu != -1):
            var = var.cuda(gpu)
        l = var.size(0)
        recipe_tensors.append((l + random.random() * 0.0001, var)) #include lengths to sort by, random perturbation to avoid same-length collisions

    #Sort and extract tensors only
    recipe_tensors = [ele[1] for ele in sorted(recipe_tensors, reverse = True)]
    recipe_tensors = torch.nn.utils.rnn.pad_sequence(recipe_tensors).permute(1,0)
    return recipe_tensors




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
parser.add_argument('--gpu', action='store_true', default = False,
                    help='use GPU')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--load', type=str, default=None,
                    help='path to load existing model from')
parser.add_argument('--prime_str', type=str, default="chocolate",
                    help='String to prime recipe generation with')
parser.add_argument('--contains_str', type=str, default=None,
                    help='String filter generated recipes for(e.g. an ingredient)')
parser.add_argument('--n_to_generate', type=int, default=10,
                    help='Number of recipes to generate using prime_str')
parser.add_argument('--print_every', type=int, default=100,
                    help='Frequency to test/output stats during training')
parser.add_argument('--max_length', type=int, default=100,
                    help='Maximum recipe length')
parser.add_argument('--instructions', action='store_true', default = False,
                    help='Use recipe files containing both ingredients and instructions versus only ingredients. This makes the task much more difficult.')
parser.add_argument('--instr_embedding', type=str, default='./glove_recipe_instr_vectors_100.txt',
                    help='Path of GloVe embedding definition file for instructions')
parser.add_argument('--concat', type=str, default=None,
                    help='Clean and concatenate recipe files for embedding processing, then exit')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

gpu = -1
if(args.gpu):
    gpu = 0

recipe_dir = args.data

recipe_files = [os.path.join(recipe_dir, name) for name in os.listdir(recipe_dir) if "_ingredients.txt" in name]
if(args.instructions):
    instr_recipe_files = [name.replace("_ingredients.txt", "_instructions.txt") for name in recipe_files]

if(args.concat is not None):
    concat_recipes(recipe_files, args.concat)
    if(args.instructions):
        concat_recipes(instr_recipe_files, args.concat + "_instructions")
    print("Recipe files concatenated, now generate an embedding with GloVe")
    exit()


print("Loading embedding...")

embedding, word_2_ind, ind_2_word = load_glove_embedding(args.embedding, args.emsize)
if(args.instructions):
    instr_embedding, instr_word_2_ind, instr_ind_2_word = load_glove_embedding(args.instr_embedding, args.emsize)

print("Embedding loaded.")

#Define RNN model
class recipeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, embedding, tie_weights = False, drop_freq = 0.0):

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

    def init_hidden(self, batch_size, gpu = -1):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        if(gpu != -1):
            h = h.cuda(gpu)
            c = c.cuda(gpu)
        return (h,c)

#Attentional RNN model for generating instructions given ingredients
class ingr2instrRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, ingr_embedding, instr_embedding, drop_freq = 0.0, max_length = 100):
        super(ingr2instrRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.max_length = max_length

        self.drop_ingr = nn.Dropout(drop_freq)

        #Encoder stuff for ingredients
        self.encoder_ingr = ingr_embedding
        self.rnn_ingr = nn.LSTM(input_size, hidden_size, n_layers, batch_first = True, dropout = drop_freq)

        #Decoder stuff for instructions
        self.encoder_instr = instr_embedding
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.max_length, self.hidden_size)
        self.drop_instr = nn.Dropout(drop_freq)

        self.rnn_instr = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first = True, dropout = drop_freq)

        self.out = nn.Linear(hidden_size, output_size)

    def forward_encoder(self, input, hidden):
        encoded = self.encoder_ingr(input)
        encoded = self.drop_ingr(encoded)
        if(len(encoded.size()) < 3):
            encoded = encoded.unsqueeze(0)
        #print(encoded, hidden)
        output, hidden = self.rnn_ingr(encoded, hidden)
        return output, hidden

    def forward_decoder(self, input, hidden, encoder_outputs):
        output = self.encoder_instr(input)
        output = self.drop_instr(output)

        attn_weights = F.softmax(
            self.attn(torch.cat((output, hidden), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((output, attn_applied), 1)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, hidden = self.rnn_instr(output, hidden)

        output = self.out(output)

        return output, hidden, attn_weights

    def init_hidden(self, batch_size, gpu = -1):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        if(gpu != -1):
            h = h.cuda(gpu)
            c = c.cuda(gpu)
        return (h,c)



#Let's load and clean our recipe dataset into a list of tensors
if(args.load is None):

    recipe_tensors = load_recipes(recipe_files, word_2_ind, args.max_length, gpu)
    if(args.instructions):
        #max length here needs to be very large so that we load instructions for every recipe loaded above
        #TODO prune ingredients based on instructions
        instr_recipe_tensors = load_recipes(instr_recipe_files, word_2_ind, args.max_length, gpu)

    print("Loaded " + str(recipe_tensors.size(0)) + " recipes")

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
    len(word_2_ind.keys()),
    n_layers = args.nlayers,
    embedding = embedding,
    tie_weights = args.tied,
    drop_freq = args.dropout
)
if(args.gpu):
    recipe_model = recipe_model.cuda(gpu)
opt = torch.optim.Adam(recipe_model.parameters(), lr=args.lr)
criterion = F.cross_entropy

#Initialize instruction attentional RNN
instr_model = None
if(args.instructions):
    instr_model = ingr2instrRNN(
        args.emsize,
        args.nhid,
        len(instr_word_2_ind.keys()),
        args.nlayers,
        embedding,
        instr_embedding,
        args.dropout,
        args.max_length
    )
    print(instr_model)
    if(args.gpu):
        instr_model = instr_model.cuda(gpu)
    instr_opt = torch.optim.Adam(instr_model.parameters(), lr=args.lr)

#pre-allocated Tensor to hold current batch
if(args.load is None):
    curr_batch = torch.LongTensor(args.batch_size, recipe_tensors.size(1))
    if(args.gpu):
        curr_batch = curr_batch.cuda(gpu)
    if(args.instructions):
        instr_curr_batch = torch.LongTensor(args.batch_size, instr_recipe_tensors.size(1))
        if(args.gpu):
            instr_curr_batch = instr_curr_batch.cuda(gpu)

#Train by predicting next word given current word and hidden state
#Batches currently handled by running on multiple sequences in series, accumulating gradient updates before calling opt. 
#May not be the most efficient solution...
def train(inputs, instr_inputs = None):
    hidden = recipe_model.init_hidden(args.batch_size, gpu)
    recipe_model.zero_grad()

    loss = 0

    for b in range(args.batch_size):
        rand_ind = int(math.floor(random.random() * inputs.size(0)))
        curr_seq = inputs[rand_ind]        
        curr_batch[b] = curr_seq
        #Indices should be matched up, I think...
        if(instr_inputs is not None):
            instr_curr_batch[b] = instr_inputs[rand_ind]
        
    ind = 0
    loss = 0
    instr_loss = 0
    while ind + 1 < curr_batch.size(1):

        output, hidden = recipe_model(curr_batch[:,ind:ind+1], hidden)
        loss += criterion(output, curr_batch[:,ind+1])
        ind += 1

        print(curr_batch[:,ind:ind+1])
    #targets = Variable(curr_seq.data.new().resize_(curr_seq.size()))
    #targets[0,0:targets.size(1) - 1] = curr_seq[0,1:]
    #targets[0,targets.size(1) - 1] = word_2_ind["<eol>"]

    #output, hidden = recipe_model(curr_seq, hidden)
    #loss += criterion(output.squeeze(), targets.squeeze())
    loss.backward()
    #Clip gradients in case of exploding gradients
    torch.nn.utils.clip_grad_norm(recipe_model.parameters(), args.clip)
    opt.step()

    #Now train instruction generator
    #First, encode ingredients
    #TODO This could be merged with generation above?
    if(instr_inputs is not None):

        instr_model.zero_grad()

        encoder_outputs = torch.zeros(args.max_length, args.nhid)
        encoder_hidden = instr_model.init_hidden(args.batch_size)
        if(args.gpu):
            encoder_outputs.cuda(gpu)

        for ingr in range(curr_batch.size(1)):
            encoder_output, encoder_hidden = instr_model.forward_encoder(curr_batch[:,ingr:ingr+1], encoder_hidden)
            encoder_outputs[ingr] = encoder_output
        #now, attend encoding and decode/generate
        ind = 0
        instr_hidden = encoder_hidden
        while ind + 1 < instr_curr_batch.size(1):
            instr_output, instr_hidden, instr_attention = instr_model.forward_decoder(instr_curr_batch[:,ind:ind+1], instr_hidden, encoder_outputs)
            instr_loss += criterion(instr_output, instr_curr_batch[:,ind+1])
        
        instr_loss.backward()
        instr_opt.step()
            

    return loss.item() / curr_batch.size(1), instr_loss.item() / instr_curr_batch.size(1)


#===Generate Text

def generate(recipe_model, instr_model, prime_str='a', predict_len=60, temperature=0.8, cuda=False):

    #parse the priming string
    tokens = []
    prime_tok = clean_line(prime_str).split(" ")
    for tok in prime_tok:
        if(tok in word_2_ind.keys()):
            tokens.append(word_2_ind[tok])
        else: #unknown word
            tokens.append(word_2_ind["<unk>"])
    prime_input = torch.LongTensor(tokens).unsqueeze(0)

    hidden = recipe_model.init_hidden(1, gpu)
    if cuda:
        prime_input = prime_input.cuda()
    predicted = prime_str

    for p in range(len(prime_tok) - 1):
        _, hidden = recipe_model(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    outp = torch.zeros(1, predict_len)
    if cuda:
        outp = outp.cuda(gpu)
    
    for p in range(predict_len):
        output, hidden = recipe_model(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        if(instr_model is not None):
            outp[:,p] = instr_word_2_ind[ind_2_word[top_i]]

        # Add predicted character to string and use as next input
        predicted_char = ind_2_word[top_i.item()]
        if(predicted_char == "<eos>"):
            break
        predicted += " " + predicted_char.replace("<eol>","\n")
        inp = torch.LongTensor([word_2_ind[predicted_char]]).unsqueeze(0)
        if cuda:
            inp = inp.cuda(gpu)

    #Now generate instructions to go with ingredients
    if(instr_model is not None):
        instr_hidden = instr_mode.init_hidden(1, gpu)
        encoder_outputs = torch.zeros(args.max_length, args.nhid)
        if cuda:
            encoder_outputs = encoder_outputs.cuda(gpu)
        #encode
        for p in range(predict_len):
            encoder_outputs[p], instr_hidden = instr_model.forward_encoder(outp[:,p:p+1], instr_hidden)
        #decode
        #TODO add instruction prime string
        decoder_input = prime_input[:,-1]
        for p in range(predict_len):
            output, instr_hidden = instr_model.forward_decoder(decoder_input, instr_hidden, encoder_outputs)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted_char = instr_ind_2_word[top_i.item()]
            if(predicted_char == "<eos>"):
                break
            predicted += " " + predicted_char.replace("<eol>","\n")
            decoder_input = torch.LongTensor([instr_word_2_ind[predicted_char]]).unsqueeze(0)
            if cuda:
                decoder_input = decoder_input.cuda(gpu)


            
        

    return predicted


#===Train and run!        

print(recipe_model)

start = time.time()
all_losses = []
loss_avg = 0
instr_loss_avg = 0

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
        instr_input = None
        if(args.instructions):
            instr_input = instr_recipe_tensors
            instr_model.train(True)
        recipe_model.train(True)
        loss, instr_loss = train(recipe_tensors, instr_input)
        loss_avg += loss
        instr_loss_avg += instr_loss

        if epoch % args.print_every == 0:
            recipe_model.train(False)
            if(args.instructions):
                instr_model.train(False)
            print('[%s (%d %d%%) %.4f %.4f]' % (time_since(start), epoch, epoch / args.iters * args.batch_size, loss_avg / args.print_every, instr_loss_avg / args.print_every))
            print('loss: ', loss, instr_loss)
            print(generate(recipe_model, instr_model, args.prime_str, args.max_length, 0.8, cuda=args.gpu) + "\n")
            loss_avg = 0

    #Save model to file
    torch.save(recipe_model, args.save + ".model")
    torch.save(recipe_model.state_dict(), args.save + ".weights")
    if(args.instructions):
        torch.save(instr_model, args.save + "_instr.model")
        torch.save(instr_model.state_dict(), args.save + "_instr.weights")

recipe_model.train(False)
if(args.instructions):
    instr_model.train(False)
n_sampled = 0
while n_sampled < args.n_to_generate:
    recipe = generate(recipe_model, instr_model, args.prime_str, args.max_length, temperature= 0.8, cuda=args.gpu)
    if(args.contains_str is not None):
        if(args.contains_str in recipe):
            print(recipe)
            n_sampled += 1
    else:
        print(recipe)
        n_sampled += 1


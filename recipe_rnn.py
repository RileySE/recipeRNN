#Learn to model recipes and generate new ones via lstm

#TODO Add attention over previous decoder outputs(maybe for ingredients too?) based on paulus, xiong, socher, 2017

#TODO Add REINFORCE updates on non-teacher-forcing network outputs- incentivise outputting words which make the network better at predicting future words as well as matching the current target word.

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
    return line.replace("'","").replace(","," ,").replace(":","").replace(";","").replace("(","( ").replace(")"," )").replace("&#39", "").replace("\n", " <eol>").replace(".", " .").lower()

#Clean and concatenate recipe files
def concat_recipes(recipe_files, outfile_name):
    allrecipes_file = open(outfile_name, "w")
    for recipe in recipe_files:
        #print(recipe)
        allrecipes_file.write("<sos> ")
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
    recipes_loaded = []
    for recipe in recipe_files:
        #print(recipe)
        tokens = [word_2_ind["<sos>"]]
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
        #recipe_tensors.append((l + len(recipe_tensors) * 0.000001, var)) #include lengths to sort by
        recipe_tensors.append(var)
        recipes_loaded.append(recipe)

    #Sort and extract tensors only
    #recipe_tensors = [ele[1] for ele in sorted(recipe_tensors, reverse = True)]
    #recipe_tensors = torch.nn.utils.rnn.pad_sequence(recipe_tensors).permute(1,0)
    return recipe_tensors, recipes_loaded




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
parser.add_argument('--instr_max_length', type=int, default=200,
                    help='Max length for instruction sequences')
parser.add_argument('--instr_load', type=str, default=None,
                    help='Path to saved weights for instruction generation network')
parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0,
                    help='Fraction of the time to use teacher forcing')
parser.add_argument('--temperature', type=float, default=0.8,
                    help='Temperature to use for sampling from the RNN distribution')
parser.add_argument('--concat', type=str, default=None,
                    help='Clean and concatenate recipe files for embedding processing, then exit')
parser.add_argument('--save_every', type=int, default=2000,
                    help='Frequency to save out network weights during training')
parser.add_argument('--resume', action='store_true', default = False,
                    help='If --load is specified, continue training rather than generating samples')
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
        self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first = True, dropout = drop_freq)
          
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
        #c = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        if(gpu != -1):
            h = h.cuda(gpu)
            #c = c.cuda(gpu)
        #return (h,c)
        return h

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
        self.rnn_ingr = nn.GRU(input_size, hidden_size, n_layers, batch_first = True, dropout = drop_freq, bidirectional = True)

        #Decoder stuff for instructions
        self.encoder_instr = instr_embedding
        self.attn = nn.Linear(self.hidden_size * (n_layers*2 + 1), self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.drop_instr = nn.Dropout(drop_freq)

        self.rnn_instr = nn.GRU(hidden_size, hidden_size * 2, n_layers, batch_first = True, dropout = drop_freq)

        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward_encoder(self, input, hidden):
        encoded = self.encoder_ingr(input)
        encoded = self.drop_ingr(encoded)
        if(len(encoded.size()) < 3):
            encoded = encoded.unsqueeze(0)
        #print(encoded.size(), hidden.size())
        output, hidden = self.rnn_ingr(encoded, hidden)
        return output, hidden

    def forward_decoder(self, input, hidden, encoder_outputs):
        output = self.encoder_instr(input)
        output = self.drop_instr(output)
        concat = torch.cat((output, hidden.view(hidden.size(1), 1, -1)), 2)
        concat = concat.reshape(concat.size(0), -1)
        attn_weights = self.attn(concat)
        attn_weights = F.softmax(attn_weights, dim=1)
        #print(attn_weights.unsqueeze(1).size(), encoder_outputs.size())
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        #print(attn_applied.size(), output.size())

        output = torch.cat((output, attn_applied), 2)
        output = self.attn_combine(output)
        output = F.relu(output)

        output, hidden = self.rnn_instr(output, hidden)

        output = self.out(output.squeeze())

        return output, hidden, attn_weights

    def init_hidden(self, batch_size, gpu = -1):
        #For bidirectional encoder
        h = torch.zeros(self.n_layers * 2, batch_size, self.hidden_size)
        #c = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        if(gpu != -1):
            h = h.cuda(gpu)
            #c = c.cuda(gpu)
        #return (h,c)
        return h



#Let's load and clean our recipe dataset into a list of tensors
if(args.load is None or args.resume):

    if(args.instructions):
        #Filter loaded ingredients files based on instructions since these are more likely to hit the length cap
        instr_recipe_tensors, instr_recipes_loaded = load_recipes(instr_recipe_files, instr_word_2_ind, args.instr_max_length, gpu)
        to_load = [name for name in recipe_files if name.replace("_ingredients.txt","_instructions.txt") in instr_recipes_loaded]
        length = 1000000
    else:
        to_load = recipe_files
        length = args.max_length
    recipe_tensors, recipes_loaded = load_recipes(to_load, word_2_ind, length, gpu)


    #print(recipes_loaded[len(recipes_loaded)-1], instr_recipes_loaded[len(instr_recipes_loaded)-1])
    print("Loaded " + str(len(recipe_tensors)) + " recipes")
    print("Rejected " + str(len(recipe_files) - len(recipe_tensors)) + " recipes")

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
if(args.load is None or args.resume):
    curr_batch = torch.LongTensor(args.batch_size, args.max_length)
    if(args.gpu):
        curr_batch = curr_batch.cuda(gpu)
    if(args.instructions):
        instr_curr_batch = torch.LongTensor(args.batch_size, args.instr_max_length)
        #Mask to block padding values for encoder representation
        curr_batch_mask = torch.zeros_like(curr_batch).float()
        if(args.gpu):
            instr_curr_batch = instr_curr_batch.cuda(gpu)
            curr_batch_mask = curr_batch_mask.cuda(gpu)

#Train by predicting next word given current word and hidden state
#Batches currently handled by running on multiple sequences in series, accumulating gradient updates before calling opt. 
#May not be the most efficient solution...
def train(inputs, instr_inputs = None):
    hidden = recipe_model.init_hidden(args.batch_size, gpu)
    recipe_model.zero_grad()

    loss = 0

    curr_batch.fill_(word_2_ind["<eos>"])
    if(instr_inputs is not None):
        curr_batch_mask.fill_(0)
        instr_curr_batch.fill_(instr_word_2_ind["<eos>"])
    for b in range(args.batch_size):
        rand_ind = int(math.floor(random.random() * len(inputs)))
        curr_seq = inputs[rand_ind]
        #print(curr_seq.size())
        max_ind = min(curr_seq.size(0), args.max_length)
        curr_batch[b,:max_ind] = curr_seq[:max_ind]
        #curr_batch[b] = curr_seq
        #Indices should be matched up, I think...
        if(instr_inputs is not None):
            curr_batch_mask[b,:max_ind].fill_(1)
            instr_curr_seq = instr_inputs[rand_ind]
            max_ind = min(instr_curr_seq.size(0), args.instr_max_length)
            instr_curr_batch[b,:max_ind] = instr_curr_seq[:max_ind]

        #for ele in curr_seq:
        #    print(ind_2_word[ele.item()])
        #for ele in instr_inputs[rand_ind]:
        #    print(instr_ind_2_word[ele.item()])
        
    ind = 0
    loss = 0
    #If we are not using teacher forcing, we should provide the output of the ingredient network to the instruction network encoder
    ingr_outp = torch.zeros(args.batch_size, curr_batch.size(1)).long()
    if(args.gpu):
        ingr_outp = ingr_outp.cuda(gpu)
    ingr_outp.fill_(word_2_ind["<eos>"])
    
    #Select use of teacher forcing randomly
    tf_choice = random.random()
    if(tf_choice < args.teacher_forcing_ratio):
        while ind + 1 < curr_batch.size(1):
            output, hidden = recipe_model(curr_batch[:,ind:ind+1], hidden)
            loss += criterion(output, curr_batch[:,ind+1])
            ind += 1
    #Use previous prediction as input
    else:
        ingr_input = curr_batch[:,ind:ind+1]
        ingr_outp[:,ind] = ingr_input.squeeze()
        while ind + 1 < curr_batch.size(1):
            output, hidden = recipe_model(ingr_input, hidden)
            loss += criterion(output, curr_batch[:,ind+1])
            ind += 1
            topv, topi = output.topk(1)
            #Need to detach to not backprop across inputs
            ingr_input = topi.detach()
            ingr_outp[:,ind] = ingr_input.squeeze()


    #targets = Variable(curr_seq.data.new().resize_(curr_seq.size()))
    #targets[0,0:targets.size(1) - 1] = curr_seq[0,1:]
    #targets[0,targets.size(1) - 1] = word_2_ind["<eol>"]

    #output, hidden = recipe_model(curr_seq, hidden)
    #loss += criterion(output.squeeze(), targets.squeeze())
    loss /= curr_batch.size(1)
    loss.backward()
    #Clip gradients in case of exploding gradients
    torch.nn.utils.clip_grad_norm(recipe_model.parameters(), args.clip)
    opt.step()

    #Now train instruction generator
    #First, encode ingredients
    #TODO This could be merged with generation above?
    instr_loss = torch.tensor([0])
    if(instr_inputs is not None):
        instr_loss = 0
        instr_model.zero_grad()

        encoder_outputs = torch.zeros(args.batch_size, args.max_length, args.nhid)
        encoder_hidden = instr_model.init_hidden(args.batch_size, gpu)
        if(args.gpu):
            encoder_outputs = encoder_outputs.cuda(gpu)

        #Use ingredient network output if not teacher forcing
        if(tf_choice < args.teacher_forcing_ratio):
            encoder_input = curr_batch
        else:
            encoder_input = ingr_outp

        encoder_output, encoder_hidden = instr_model.forward_encoder(encoder_input, encoder_hidden)
        encoder_outputs = encoder_output
        #for ingr in range(curr_batch.size(1)):
        #    encoder_output, encoder_hidden = instr_model.forward_encoder(encoder_input[:,ingr:ingr+1], encoder_hidden)
        #    encoder_outputs[:,ingr,:] = encoder_output.squeeze()
        #now, attend encoding and decode/generate
        ind = 0
        #For bidirectional encoder, concat the hidden states for both directions and feed into decoder
        #Ordering of layer hidden states should be 1-1,2-1,3-1,1-2,2-2,3-2 I think...
        instr_hidden = torch.cat([encoder_hidden[:instr_model.n_layers], encoder_hidden[instr_model.n_layers:]], 2)
        #First, mask the encoder output to block padding values
        #Mask is batchXmax_len, need to expand to batchXmax_lenXhsize
        encoder_outputs = encoder_outputs.mul(curr_batch_mask.expand(encoder_outputs.size(2),-1,-1).permute(1,2,0))

        #Randomly choose to either use teacher forcing(GT input for previous word) or the network's prediction
        if(tf_choice < args.teacher_forcing_ratio):
            while ind + 1 < instr_curr_batch.size(1):
                instr_output, instr_hidden, instr_attention = instr_model.forward_decoder(instr_curr_batch[:,ind:ind+1], instr_hidden, encoder_outputs)
                instr_loss += criterion(instr_output, instr_curr_batch[:,ind+1])
                ind += 1
        #Use previous network prediction as input
        else:
            instr_input = instr_curr_batch[:,ind:ind+1]
            while ind + 1 < instr_curr_batch.size(1):
                instr_output, instr_hidden, instr_attention = instr_model.forward_decoder(instr_input, instr_hidden, encoder_outputs)
                instr_loss += criterion(instr_output, instr_curr_batch[:,ind+1])
                ind += 1
                #Get prediction
                #TODO maybe make this probabilistic?
                topv, topi = instr_output.topk(1)
                #Need to detach to not backprop across inputs
                instr_input = topi.detach()
                #TODO handle when the network outputs EOS?
        
        instr_loss /= instr_curr_batch.size(1)
        instr_loss.backward()
        torch.nn.utils.clip_grad_norm(instr_model.parameters(), args.clip)
        instr_opt.step()
            

    return loss.item(), instr_loss.item()


#===Generate Text

def generate(recipe_model, instr_model, prime_str='a', predict_len=60, temperature=0.8, cuda=False):

    outp = torch.zeros(1, predict_len).long()
    outp.fill_(word_2_ind["<eos>"])
    outp_ind = 0

    #parse the priming string
    tokens = [word_2_ind["<sos>"]]
    prime_tok = clean_line(prime_str).split(" ")
    for tok in prime_tok:
        if(tok in word_2_ind.keys()):
            tokens.append(word_2_ind[tok])
            outp[:,outp_ind] = word_2_ind[tok]
            outp_ind += 1
        else: #unknown word
            tokens.append(word_2_ind["<unk>"])
            outp[:,outp_ind] = word_2_ind["<unk>"]
            outp_ind += 1

    prime_input = torch.LongTensor(tokens).unsqueeze(0)

    hidden = recipe_model.init_hidden(1, gpu)
    if cuda:
        prime_input = prime_input.cuda(gpu)
    predicted = prime_str

    for p in range(len(prime_tok) - 1):
        _, hidden = recipe_model(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]

    if cuda:
        outp = outp.cuda(gpu)
    
    for p in range(predict_len):
        output, hidden = recipe_model(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        #top_k, top_i = output.data.topk(1)

        if(instr_model is not None and outp_ind < outp.size(1)):
            outp[:,outp_ind] = top_i.item()
            outp_ind += 1

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
        encoder_hidden = instr_model.init_hidden(1, gpu)
        #encoder_outputs = torch.zeros(1, args.max_length, args.nhid * 2)
        #if cuda:
        #    encoder_outputs = encoder_outputs.cuda(gpu)
        #encode
        encoder_outputs, encoder_hidden = instr_model.forward_encoder(outp, encoder_hidden)
        instr_hidden = torch.cat([encoder_hidden[:instr_model.n_layers], encoder_hidden[instr_model.n_layers:]], 2)

        #for p in range(outp.size(1)):
            #print(outp.size(), outp[:,p])
       #     encoder_outputs[0,p], instr_hidden = instr_model.forward_encoder(outp[:,p:p+1], instr_hidden)
        #decode
        decoder_input = torch.LongTensor([instr_word_2_ind["<sos>"]]).unsqueeze(0)
        if cuda:
            decoder_input = decoder_input.cuda(gpu)
        for p in range(args.instr_max_length):

            output, instr_hidden, instr_attention = instr_model.forward_decoder(decoder_input, instr_hidden, encoder_outputs)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            #top_k, top_i = output.data.topk(1)

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

if(args.instr_load is not None):
    print("Loading instruction weights from file...")
    instr_model.load_state_dict(torch.load(args.instr_load))

if(args.load is not None):
    print("Loading weights from file...")
    weights_filename = args.load
    recipe_model.load_state_dict(torch.load(weights_filename))

if(args.load is None or args.resume):
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
            print(generate(recipe_model, instr_model, args.prime_str, args.max_length, args.temperature, cuda=args.gpu) + "\n")
            loss_avg = 0
            instr_loss_avg = 0

        if epoch % args.save_every == 0:
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
    recipe = generate(recipe_model, instr_model, args.prime_str, args.max_length, temperature= args.temperature, cuda=args.gpu)
    if(args.contains_str is not None):
        accept = True
        for tok in args.contains_str.split(" "):
            if(not tok in recipe):
                accept = False
        if(accept):
            print(recipe)
            n_sampled += 1
    else:
        print(recipe)
        n_sampled += 1


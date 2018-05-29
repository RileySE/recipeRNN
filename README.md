# recipeRNN

This is a small project to develop an RNN language model to generate recipes(for food). Included is the code needed to scrape training data `get_pages.py` and `parse_html.py` as well as for training the model itself `recipe_rnn.py`

## Data
You can scrape html pages using the following(e.g. to get recipes 10000 to 20000):

```
python get_pages.py 10000 20000
```

Then, parse the raw html into a usable text file using the following hacky script:

```
python parse_html.py recipe_download_dir/
```

There is also a flag in `parse_html.py` to include instructions as well as ingredients, but these are a harder language modeling problem to solve and are not well handled currently.

Regardless, this generates a set of `*_ingredients.txt` files containing the recipe from each html page.

## Embedding

You will need to train a word embedding before training the RNN. I used the [GloVe repo](https://github.com/stanfordnlp/GloVe) and trained a GloVe embedding on the recipe corpus above, length of 100.

Before running GloVe, run the following to clean the text files and concatenate them for input to GloVe:

```
python recipe_rnn.py --concat recipe_concat.txt
```

## Training

Once you've got the GloVe embedding ready, you can train the RNN like so:

```
python recipe_rnn.py
```

There are a number of default parameters and filepath arguments defined, and you may want to define them differently, use the `--help` flag to see all arguments.

To generate recipes using the trained RNN, add `--load model.pt.weights` to the command to load the pre-trained weights.


### Acknowledgements

This project adapts code from Sebastian Seung's [COS 485 Neural Networks: Theory and Applications](https://cos485.github.io/) course, as well as the [Pytorch word-level language modeling RNN example](https://github.com/pytorch/examples/tree/master/word_language_model). 

Further, I'm grateful for the existence of online recipe websites like Allrecipes.com, without which this project would not be possible.
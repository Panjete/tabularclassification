Current Trends:

## Ideas

* Regularise, bitch!

* Try Balancing all classes by mixing text of all smaller classes, and creating new sentences
* Characters not human readable "\u00e5", are correctly classified by python as a single character!
* Misclassifications happen mostly by incorrectly claiming english

* Easy - Correction by first reducing capital standard letter to small ones  - hopefully helps increase frequencies

* ngram_range 

* analyzer : {'word', 'char', 'char_wb'} or callable, default='word'
    Whether the feature should be made of word n-gram or character n-grams. Option 'char_wb' creates character n-grams only from text inside word boundaries; n-grams at the edges of words are padded with space.

* Try XGBoost - said to auto-correct class size differences

* Resample with different proportions of english - tune to whatever model works best

* can actually try min_df - sounds cool - will remove any tokens that occur less than given (x) number of times. for example - if "helonamaste" just occured once in spanish - maybe isn't representative enough of spanish. 

* try tfidfvectorizer

* Try keeping closer languages closer. The following is the order used. 

1. English
1. German
1. Swedish
1. French
1. Italian
1. Spanish
1. Portuguese
1. Bengali
1. Hindi
1. Marathi
1. Tamil
1. Malayalam
1. Kannada

This order has been kept because :- 
    * English, German, Swedish, French, Italian, and Spanish are all Indo-European languages, specifically belonging to the Germanic (English, German, Swedish), Romance (French, Italian, Spanish), or both (French and Italian).
    * Portuguese is also a Romance language and shares similarities with Spanish.
    * Bengali and Marathi are both Indo-Aryan languages, with Bengali being more closely related to Hindi.
    * Tamil, Malayalam, and Kannada are Dravidian languages, with Tamil and Malayalam being particularly close due to geographical proximity and historical connections. Kannada is also a Dravidian language but is generally considered to be more distinct from Tamil and Malayalam.

* Ensemble approaches?

* search for feature engineering in language classification tasks

* Unless otherwise mentioned, the model being ablated is LR with 200k max parameters
## NB 

#### Simple 

Accuracy       =  0.981
Macro F1 score =  0.959
Micro F1 score =  0.969


#### Challenge

Accuracy       =  0.782
Macro F1 score =  0.776
Micro F1 score =  0.703


## LR

#### Simple

Accuracy       =  0.973
Macro F1 score =  0.931
Micro F1 score =  0.957

keeping languages closer :

Accuracy       =  0.974
Macro F1 score =  0.932
Micro F1 score =  0.957


#### Challenge

Accuracy       =  0.811
Macro F1 score =  0.781
Micro F1 score =  0.730

keeping languages closer :

Accuracy       =  0.811
Macro F1 score =  0.779
Micro F1 score =  0.728

with tfidf vectoriser, bigram instead:

Accuracy       = 0.798
Macro F1 score = 0.748
Micro F1 score = 0.712

with char encodings, 800k, pentagrams:

Accuracy       = 0.913
Macro F1 score = 0.893
Micro F1 score = 0.873

## RF (21 estimators)

vanilla vs 400k bigrams
#### Simple 

Accuracy       =  0.966
Macro F1 score =  0.895
Micro F1 score =  0.944


#### Challenge

Accuracy       =  0.814
Macro F1 score =  0.787
Micro F1 score =  0.736


### LR (Bigrams, upto 80k, 100k, 120k, 200k, 2000k features)

#### Simple 
Accuracy       =  0.972, 0.973, 0.975, 0.975, 0.975
Macro F1 score =  0.904, 0.918, 0.930, 0.938, 0.932
Micro F1 score =  0.954, 0.956, 0.958, 0.960, 0.959

#### Challenge

Accuracy       =  0.784, 0.7931, 0.804, 0.814, 0.819
Macro F1 score =  0.704, 0.7302, 0.752, 0.778, 0.786
Micro F1 score =  0.694, 0.7058, 0.719, 0.773, 0.741


### LR (Trigrams, 200k, 400k, 800k, 1600k features)

#### Simple 
Accuracy       =  0.974, 0.975, 0.975, 0.975
Macro F1 score =  0.927, 0.936, 0.938, 0.938
Micro F1 score =  0.958, 0.959, 0.960, 0.960

#### Challenge

Accuracy       =  0.798, 0.805, 0.809, 0.813
Macro F1 score =  0.743, 0.761, 0.773, 0.778
Micro F1 score =  0.713, 0.722, 0.728, 0.733


### LR 3000 iterations - char_wb (80k - 400k) same for both

#### Simple

Accuracy       = 0.871
Macro F1 score = 0.848
Micro F1 score = 0.794

#### Challenge

Accuracy       = 0.747, 
Macro F1 score = 0.685
Micro F1 score = 0.641

## Removing numbers from dataset classified as tokens

without vs with vs (prune numbers and underscores):

Accuracy       =  0.809 vs 0.801 vs 0.796
Macro F1 score =  0.773 vs 0.749 vs 0.742
Micro F1 score =  0.728 vs 0.717 vs 0.710

All the below stuff is with tfidf, unless specified
#### char embeddings, pentagrams
Simple:

Accuracy       = 0.985
Macro F1 score = 0.982
Micro F1 score = 0.975

Challenge:

Accuracy       = 0.913
Macro F1 score = 0.893
Micro F1 score = 0.873


####  char_wb Embeddings, pentagrams
#### Same for L2 regulariser
Simple:

Accuracy       = 0.985
Macro F1 score = 0.983
Micro F1 score = 0.976

Challenge:

Accuracy       = 0.914
Macro F1 score = 0.900
Micro F1 score = 0.874

Hence, fixing char_wb

Now try:
    septagrams, failed
    countvectoriser, on it
    elastic, next - train file written

####  char_wb Embeddings, septagrams

Accuracy       = 0.984
Macro F1 score = 0.981
Micro F1 score = 0.974

Accuracy       = 0.910
Macro F1 score = 0.887
Micro F1 score = 0.868

####  char_wb Embeddings, pentagrams, countvec

Did not converge

#### char
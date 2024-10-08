## Tabular QA

The ability to understand and process tabular data is crucial in many fields, including finance, healthcare, and science. With the vast amount of information stored in tables, there’s a growing need for models that can accurately interpret and answer questions based on tabular data. Further, tables are 2 dimensional and direct use of sequence to sequence language models may not be the optimal approach. This repository provides implementations of a myraid of techniques and architectures for the task of Tabular Question Answering, focusing on classifying table cells that contribute to answering a given question.

## To Run

1. Set appropriate configurations in `configs.py`.
2. Run `bash run_model.sh <trainfile> <valfile>` to train the models.
3. Run `bash run_models.sh <testfile> <outputfile>` to query results from the trained models.
4. Run `evalresults.py <goldlabelsfile> <outputfile>` to obtain scores on the quality of results.


## Architectural Details

Since there is a single correct column per table, let us decompose the task into finding the column and finding the row separately.

### Column Problem 

Options : per-column-score vs all-column-at-once Architecture

    Approach 1 : generate score for column by passing <question> + <SEP> + <column> through the model
    Approach 2 : generate index for column by passing 
                        <question> + <SEP> + <col1> + [SEP2] + <col2> + <SEP2> + <col3> + ... through the model

For the task of column prediction, I discovered that while both the approaches were theoretically equivalent,
the model was having a hard time predicting the index of the correct column in approach 2, possibly because the context of the question disappears by the time
all the columns are traversed. Thus, I go with Approach 1, using a simple BCEWithLogitsLoss() over categories 0 and 1, using 3 negative samples per question as well.

GRU Concatenated it's forward and reverse directions, so half of the entries in the 0th and -1th position are just initialisers and encode no information.
These were then shunted out, with increased the performance from ~0.80 to ~0. in the column task, and also helped learning in the row task.


### Row Problem  

Constraints : Can't load all rows at once with the question.
              Cell Elements may be arbitrarily large
              Number of columns bounded (<64)

Observation : ~23k of the 25k training samples have single correct rows. So, assuming a single correct row would not be a very bad assumption

Solution    : Develop a Score based approach, ranking the score of a particular row of being correct

Model  : Transformers -> Did not seem to be training, even after a couple of hours
         GRU          -> Satisfactory performance

Architecture : For any question, 
                    Per entry, limit the number of tokens if the total row length becomes large (32)
                    Encode the column name along with each entry of this row (because question focuses on only certain columns)
                    Further, introduce Separator1 in column name and entry content for each entry in this row
                    Introduce Separator2 between each element of the row.

                    Concatenate the first and last outputs of the bidirectional GRU, pass through and MLP
                    Label 1 if correct row, 0 otherwise, BCE Loss with output of MLP
                
                At Test time, 
                    Compute score for each row.
                    The Row with the highest score is the correct entry

#### Results : 

            COLUMNS
    GRU : 2 stacks. 1e-4. caps at 0.6126 <br>
    GRU : 2 stacks. 1e-5. caps at 0.7002 <br>
    GRU : 3 stacks. 2e-5. caps at 0.7668 <br>
    GRU : 3 stacks. 1e-4. caps at 0.2784 (doesn't converge) <br>
    GRU : 4 stacks. 2e-5. caps at 0.82   (final version) <br>

            ROWS 
    GRU : 2 stacks. 2e-5. caps at 0.197 <br>
    GRU : 4 stacks. 2e-5. caps at 0.2025 <br>
    GRU : 4 stacks. 4e-5. caps at 0.2074 <br>

#### Requirements

NLTK
gensim
PyTorch
unidecode

#### Preprocessing of text 

Tokeniser : NLTL's word_tokenize vs WordPunctTokenise()

The latter actually helped the model better by separating out punctuations, which could then be weeded out.
This dampened arbitrary discovery of new tokens, and helped better learn the alphabets that were actually there.

I also unidecode() and lowercase all tokens obtained to reduce the number of tokens. for the row problem,  I select first 30
tokens per row, and first 10 tokens per cell entry being passed to the model. At all steps, if a batch is passed to any model, I pad the inputs
as per the longest sequence in the batch

#### Variations of models 

PFA the repository 'redundant', which houses all the model I tried training and optimising. I tried training
Transformers, but the model was just not training. Maybe longer training times could have yielded results.
I tried comparing 'summaries' of questions and rows with a cosine loss as well, but to no avail.
More training time and better architecture schemes could probably have converged these models as well.

#### Trained Models :

Stored in the 'trained_models' directory. To be used by moving into the parent directory, with default configs


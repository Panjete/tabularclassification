TRAIN_COL = True
TRAIN_ROW = True

EPOCHS     = 50
EPOCHS_ROW = 20
QUESTION_ANSWER_DELIMITER = ["|"] ## Demarcates question and answer
ENTRY_ROW_WISE_STOP       = ["%"] ## Demarcates multiple elements of a row
ROW_ANSWER_DELIMITER      = ["*"] ## Demarcates Row from correct answer
COL_NAME_COL_ENTRY_DELIMITER = ["~"]
TOKEN_LIMIT = 48
ROW_TOKENS  = 64
NS          = 3

MODEL_COL_STACKS = 4   
LR_COL_MODEL     = 2e-5
MODEL_ROW_STACKS = 4
LR_ROW_MODEL     = 4e-5

EMBEDDINGS = "glove-wiki-gigaword-300" 

MAX_TOKENS_ROW   = 32
MAX_TOKENS_ENTRY = 10

IMPLIES    = "&"
IS_OF_TYPE = ["has", "type"]
import sys
from train import train
from test import test

num_keywords = len(sys.argv)

if(num_keywords == 3):
    print("Training started!")
    train(sys.argv[1], sys.argv[2]) # [trainfile] [valfile]
else:
    print("Testing started!")
    test(sys.argv[2], sys.argv[3])  # [testfile]  [predfile]



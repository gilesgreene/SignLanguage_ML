import sys
from model import train_model, evaluate_model
from preprocess import preprocess_data

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == "preprocess":
            preprocess_data()
        elif sys.argv[1] == "train":
            train_model()
        elif sys.argv[1] == "evaluate":
            evaluate_model()
        else:
            print("Invalid argument. Use 'preprocess', 'train', or 'evaluate'.")
    else:
        print("Please specify 'preprocess', 'train', or 'evaluate'.")

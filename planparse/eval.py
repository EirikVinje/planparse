import argparse



def evaluate():
    pass








if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Path to pdf file(s) to evaluate. e.g ./ex_data/200921.pdf,./ex_data/150326.pdf")
    parser.add_argument("--model", type=str, default=None, help="Path to model. e.g ./models/norwai-norwai-mistral-7b-instruct-20230414162541")
    args = parser.parse_args()

    paths = args.input.split(",")
    
    # load model

    # load data

    # generate llm data

    # evaluate


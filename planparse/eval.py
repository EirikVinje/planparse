import argparse



def evaluate():
    pass








if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Path to pdf file(s) to evaluate")
    args = parser.parse_args()

    paths = args.input.split(",")
    


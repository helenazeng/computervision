import sys
import main

def train():
    if len(sys.argv) <= 1:
        print("Error: please specify a name to save your model as.")
        print("\tExample: python train.py mymodelname")
        print("This will be the same model name used to run evaluate.py")
        return

    model_name = sys.argv[1]

    print("Creating network...")
    network = main.create_network()
    print("Loading data...")
    data = main.load_data()

    print("Training!")
    model = main.train_network(network, *data)
    model.save(model_name)

    print("Success!")
    print("Run 'python evaluate.py {}' to see the MAE".format(model_name))


if __name__ == "__main__":
    train()

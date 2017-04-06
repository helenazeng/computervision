import sys
import tflearn
import os
from main import create_network, data_root
import mahotas as mh
import numpy as np

prediction_dir = 'output'

def main(*args):
    if len(args) <= 1:
        print("Invalid usage: please specify a model file")
        print("Example: python evalue_model.py test.model")
        return

    if not check_folder_paths():
        return

    model_file = args[1]
    model = load_model(model_file)

    make_predictions(model)
    evaluate_predictions()


def load_model(model_file):
    network = create_network()
    model = tflearn.DNN(network)
    model.load(model_file)
    return model

def make_predictions(model):
    training_dir = os.path.join(data_root, 'train', 'color')

    print('Loading data...')

    images, filenames = load_images_from(training_dir)

    # remove stale output files at this point
    stale_files = (os.path.join(prediction_dir, f) for f in os.listdir(prediction_dir))
    stale_files = (f for f in stale_files if os.path.isfile(f))
    for f in stale_files:
        os.remove(f)

    print('Predicting (may take a while)...')

    counter = 0
    total_num = len(filenames)
    batch_size = 100

    images_batched = batchify(images, batch_size)
    filenames_batched = batchify(filenames, batch_size)

    for ibatch, fbatch in zip(images_batched, filenames_batched):
        predictions = model.predict(ibatch)

        for pred, fname in zip(predictions, fbatch):
            val = np.array(pred)
            mhval = mh.as_rgb(val[:,:,0], val[:,:,1], val[:,:,2])
            out_file = os.path.join(prediction_dir, fname)
            mh.imsave(out_file, mhval)

        counter += batch_size
        print('{}/{}...'.format(counter, total_num))

def evaluate_predictions():
    if data_root not in sys.path:
        sys.path.append(data_root)

    import Evaluation_script_ as EvalScript

    train_data = os.path.join(data_root, 'train', 'color')
    mae = EvalScript.evaluate(prediction_dir, train_data, None)
    print("Mean Angle Error: {}".format(mae))


def load_images_from(directory):
    filenames = [f for f in os.listdir(directory) if f.endswith('.png')]
    full_paths = (os.path.join(directory, f) for f in filenames)
    data = ( mh.imread(f) for f in full_paths )
    return data, filenames

def check_folder_paths():
    if not os.path.exists(data_root):
        print("Error: data folder '{}' does not exist".format(data_root))
        return False

    train_data = os.path.join(data_root, 'train', 'color')
    if not os.path.exists(train_data):
        print("Error: training data '{}' does not exist".format(train_data))
        return False

    train_normals = os.path.join(data_root, 'train', 'normal')
    if not os.path.exists(train_normals):
        print("Error: training normals '{}' does not exist".format(train_normals))
        return False

    if not os.path.exists(prediction_dir):
        print("*** Creating prediction directory ***")
        os.mkdir(prediction_dir)
    elif os.path.isfile(prediction_dir):
        print("Error: directory '{}' is a file! Expected it to be a directory".format(prediction_dir))
        return False

    return True

def batchify(val, batch_size):
    it = iter(val)
    done = False
    buf = [None]*batch_size
    while True:
        for i in range(batch_size):
            try:
                buf[i] = next(it)
            except StopIteration:
                done = True
                buf = buf[:i]
                if len(buf) > 0:
                    yield buf
                break

        if done:
            break

        yield buf


if __name__ == "__main__":
    main(*sys.argv)



import pickle
from sklearn.model_selection import train_test_split

TRAIN_FILE = '../data/augmented_train.pickle'

ACTIVE_IMPLEMENTATION = 'sermanet_lecun'

implementation = __import__(ACTIVE_IMPLEMENTATION, globals(), locals(), ['pipeline'], 0)


def load_train_data():
    with open(TRAIN_FILE, mode='rb') as f:
        train = pickle.load(f)

    return train_test_split(train['features'], train['labels'], test_size=0.20)


train_input, validation_input, train_output, validation_output = load_train_data()
implementation.pipeline.train(train_input, train_output, validation_input, validation_output, 43)

from data import get_data_from_files
from model import MultilayerAutoEncoder
from pathlib import Path


base_dir = str(Path().resolve().parent)


trainingFile = 'WUSTL-TRAIN.csv'
validationFile= 'WUSTL-VALIDATION.csv'
testFile=  'WUSTL-TEST-balanced.csv'


x_train, x_val, x_test, y_val, y_test, lval, ltest =  get_data_from_files(trainingFile, validationFile, testFile)

print("def = " + str(x_train.shape) )
print("def = " + str(x_val.shape) + " - " +  str(y_val.shape) )
print("def = " + str(x_test.shape) + " - " +  str(y_test.shape) )

input_dim   = x_train.shape[1]
autoencoder = MultilayerAutoEncoder(input_dim = input_dim)

autoencoder.summary()

history, threshold = autoencoder.train(x_train, x_train)

autoencoder.evaluate(x_val, y_val, threshold, lval)
autoencoder.evaluate(x_test, y_test, threshold, ltest)


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras import regularizers, initializers
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import IsolationForest

from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor

from sklearn.metrics import roc_curve
from matplotlib import pyplot
from numpy import sqrt
from numpy import argmax



np.random.seed(77145)


class MultilayerAutoEncoder():

    def __init__(self, input_dim):
        input_layer = Input(shape=(input_dim,))

        layer = Dense(36, activation='relu', kernel_initializer=initializers.RandomNormal(), activity_regularizer=regularizers.l1(10e-5))(input_layer)

        layer = Dense(8, activation='relu', kernel_initializer=initializers.RandomNormal(), activity_regularizer=regularizers.l1(10e-5))(layer)

        layer = Dense(36, activation='relu', kernel_initializer=initializers.RandomNormal(), activity_regularizer=regularizers.l1(10e-5))(layer)

        output_layer = Dense(input_dim, activation='tanh', kernel_initializer=initializers.RandomNormal(), activity_regularizer=regularizers.l1(10e-5))(layer)
        self.autoencoder = Model(inputs=input_layer, outputs=output_layer)


    def summary(self, ):
        self.autoencoder.summary()

    def train(self, x, y):

        epochs = 100
        batch_size = 1024
        validation_split = 0.1

        print('Start training.')

        opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
        self.autoencoder.compile(optimizer=opt, loss='mean_squared_error')

        history = self.autoencoder.fit(x, y,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       validation_split=validation_split,
                                       verbose=2)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


        # --- VALIDATION SET --- #
        x_val = x[x.shape[0]-(int)(x.shape[0]*validation_split):x.shape[0]-1, :]


        #---- adaptive threshold --- #
        iso = IsolationForest(n_estimators=100, contamination=0.05)
        yhat = iso.fit_predict(x_val)

        outlierPos = yhat == -1
        outlierVal = x_val[outlierPos]
        print("def = " + str(outlierVal.shape))
        outlierPred = self.autoencoder.predict(outlierVal)
        outlierMSE = np.mean(np.power(outlierVal - outlierPred, 2), axis=1)

        outIQR = np.percentile(outlierMSE, 75) - np.percentile(outlierMSE, 25)
        outUpW = np.percentile(outlierMSE, 75) + 1.5*outIQR

        inlierPos = yhat == 1
        inlierVal = x_val[inlierPos]
        print("def = " + str(inlierVal.shape))
        inlierPred = self.autoencoder.predict(inlierVal)
        inlierMSE = np.mean(np.power(inlierVal - inlierPred, 2), axis=1)

        inIQR = np.percentile(inlierMSE, 75) - np.percentile(inlierMSE, 25)
        inUpW = np.percentile(inlierMSE, 75) + 1.5*inIQR

        threshold = np.percentile(outlierMSE, 100)
        outCountPrev = ( outlierMSE[ ( outlierMSE <= threshold)]).shape[0]
        inCountPrev  = (  inlierMSE[ (inlierMSE   <= threshold)]).shape[0]

        count = 0
        sectedth = 0

        for perc in range(1, 99, 1):

            threshold = np.percentile(outlierMSE, 100-perc)

            outCountCurr = (outlierMSE[(outlierMSE <= threshold)]).shape[0]
            inCountCurr  = (inlierMSE [(inlierMSE <= threshold)]).shape[0]

            withinInCount =  inCountPrev-inCountCurr
            withinOutCount = outCountPrev-outCountCurr

            print(str(100-perc) + "   " + str(threshold) + "  "  +
                  " in_prev = " + str(inCountPrev) + " in_curr = " + str(inCountCurr) + "   "  + "(within = " + str(withinInCount) + " )" +
                  " out_prev = " + str(outCountPrev) + " out_curr = " + str(outCountCurr) + "   " +
                  " (within = " + str(withinOutCount) + " ) ... approx = " + str(outlierVal.shape[0 ]*0.01))

            inCountPrev  = inCountCurr
            outCountPrev = outCountCurr

            if ( withinInCount > withinOutCount ):
                count=count+1
                print(" ------------ ")
            if (count == 3):
                print(" ************  ")
                sectedth=threshold
                count = count + 1

        threshold=sectedth


        print("THRESH ----> " + str(threshold))


        # --- outlier computation --- #

        iso = IsolationForest(n_estimators=100, contamination=0.05)
        yhat = iso.fit_predict(x_val)

        #------------------------ #


        # ---- percentile approach ( threshold fixed) ---- #
        val_predictions = self.autoencoder.predict(x_val)
        val_mse = np.mean(np.power(x_val - val_predictions, 2),
                          axis=1)        )

        thresholdFixed = np.percentile( val_mse, 90 )
        print("THRESH (fixed)----> " + str(thresholdFixed))
        # ---
        #threshold=thresholdFixed

        df_history = pd.DataFrame(history.history)
        return df_history, threshold

    def evaluate(self, x_test, y_test, threshold, ymc):
        predictions = self.autoencoder.predict(x_test)


        # ----- reconstruction error --- #
        mse = np.mean(np.power(x_test - predictions, 2), axis=1)


        #---- multi class split --------#
        outcome = mse <= threshold

        eval = pd.DataFrame(data={'prediction': outcome, 'Class': ymc['Class'].values})


        TN = 0
        TP = 0
        FN = 0
        FP = 0


        print("")
        print("---------------")

        print(eval['Class'].value_counts())

        print("FN      TP      TN      FP        R                                Class (total)")

        classes = eval['Class'].unique()
        for c in classes:
            if c != 'BENIGN':
                A = eval[(eval['prediction'] == True) & (eval['Class'] == c)].shape[0]
                B = eval[(eval['prediction'] == False) & (eval['Class'] == c)].shape[0]


                print(str(A) + "       " + str(B) + "      -       -       " + str(
                    B / (A + B)) + "                  " + str(c) + " ( " + str(A + B) + " )")

                FN = FN + A
                TP = TP + B


            else:
                TN = eval[(eval['prediction'] == True) & (eval['Class'] == 'BENIGN')].shape[0]
                FP = eval[(eval['prediction'] == False) & (eval['Class'] == 'BENIGN')].shape[0]
                print("-       -       " + str(TN) + "       " + str(FP))


        print("FN: " + str(FN))
        print("TP: " + str(TP))
        print("TN: " + str(TN))
        print("FP: " + str(FP))

        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f1 = 2 * ((precision * recall) / (precision + recall))
        falposrat = FP / (FP + TN)

        print('R  = ', recall);
        print('P  = ', precision);
        print('F1 = ', f1);
        print('FPR=', falposrat);

        #------------------------------–#


        df_error = pd.DataFrame({'reconstruction_error': mse, 'true_class': y_test})
        print(df_error.describe(include='all'))

        plot_reconstruction_error(df_error, threshold)
        compute(df_error, threshold)
        plot_thresold(df_error.true_class, df_error.reconstruction_error, df_error, threshold)


from pathlib import Path
base_dir = str(Path().resolve().parent)

def plot_reconstruction_error(errors, threshold):
    groups = errors.groupby('true_class')
    fig, ax = plt.subplots(figsize=(15, 7))
    right = 0
    for name, group in groups:
        if max(group.index) > right: right = max(group.index)
        ax.plot(group.index, group.reconstruction_error, linestyle = '', markersize=5,label = 'Normal' if int(name)                 == 0 else 'Attack', marker ='o' if int(name) == 0 else 'v', color = 'blue' if int(name) == 0 else 'silver')              #markeredgecolor = 'black'                markersize=5,label = 'Normal' if int(name) == 0 else 'Attack', marker ='o' if int(name) == 0 else 'v', color = 'lightgreen' if int(name) == 0 else 'silver')

    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors = 'red', zorder = 100, label = 'Threshold',linewidth=3,linestyles='dashed')
    ax.semilogy()
    ax.legend(prop={'size': 20}, loc='upper right',frameon=False, ncol=3)
    plt.xlim(left = 0, right = right)
    plt.ylim(bottom=0.00001, top=100000)
    plt.title('')
    plt.yticks(fontsize=25)
    plt.xticks([])
    plt.ylabel('RE', fontsize=25)
    plt.xlabel('data points (test set)', fontsize=25)
    plt.savefig(base_dir + '/reconstruction_error.png', bbox_inches='tight', dpi=500)
    plt.show()


def compute(df_error, threshold):
    y_pred = [1 if e > threshold else 0 for e in df_error.reconstruction_error.values]
    conf_matrix = confusion_matrix(df_error.true_class, y_pred)

    tn, fp, fn, tp = conf_matrix.ravel()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * ((precision * recall) / (precision + recall))
    falposrat = fp / (fp + tn)
    trueposrat = tp / (tp + fn)

    print('R  = ', recall);
    print('P  = ', precision);
    print('F1 = ', f1);
    print('FPR=', falposrat);

    sns.heatmap(conf_matrix, xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'], annot=True, fmt='d');
    plt.title('Confusion matrix')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(base_dir + '/confusion_matrix.png', bbox_inches='tight', dpi=500)
    plt.show()
    return falposrat, trueposrat

def plot_thresold(true_class, re, df_error, threshold):
    fpr, tpr, thresholds = roc_curve(true_class, re)
    print('FPR th')
    print(fpr)
    falposrat, trueposrate = compute(df_error, threshold)
    print('FPR th 2 = ', falposrat);
    print('TPR th 2 = ', trueposrate);
    gmeans = sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    pyplot.plot([0, 1], [0, 1], color="green", lw=2, linestyle='--', label='no skill')
    pyplot.plot(fpr, tpr, color='blue', marker='.', label='supervised threshold')
    pyplot.plot(fpr[ix], tpr[ix], marker='o', markersize=15, markeredgecolor='black', linestyle='None', color='gold',
                label='best supervised threshold')
    print('TPR best')
    print(tpr[ix])
    print('FPR best')
    print(fpr[ix])
    pyplot.plot(falposrat, trueposrate, marker='X', markeredgecolor='black', markersize=15, color='red',
                linestyle='None', label='semi-supervised threshold')
    pyplot.xlabel('False Positive Rate', fontsize=25)
    pyplot.xticks(fontsize=17)
    pyplot.ylabel('True Positive Rate', fontsize=25)
    pyplot.yticks(fontsize=17)
    pyplot.legend(prop={'size': 25}, frameon=False)
    pyplot.box(on=None)
    pyplot.xticks(fontsize=25)
    pyplot.yticks(fontsize=25)
    plt.show()
    plt.show()
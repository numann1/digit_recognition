import tensorflow as tf
import numpy as np
import string
import sys
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def __init__(self):
        """This class was automatically created by the PyQt5 designer app, later I added the below few methods"""
        self.file_name = 'assets/a.jpg'
        self.digit_to_word = {y: x for x, y in zip(list(string.ascii_uppercase), range(26))}
        self.digit_to_word[-1] = '?'

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(612, 418)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.img = QtWidgets.QLabel(self.centralwidget)
        self.img.setGeometry(QtCore.QRect(0, 0, 200, 200))
        self.img.setText("")
        self.img.setPixmap(QtGui.QPixmap("assets/q_mark.png"))
        self.img.setScaledContents(True)
        self.img.setObjectName("img")
        self.descText = QtWidgets.QLabel(self.centralwidget)
        self.descText.setGeometry(QtCore.QRect(216, 52, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.descText.setFont(font)
        self.descText.setObjectName("descText")
        self.selectFile = QtWidgets.QPushButton(self.centralwidget)
        self.selectFile.setGeometry(QtCore.QRect(50, 290, 111, 51))
        self.selectFile.setObjectName("selectFile")
        self.predict = QtWidgets.QPushButton(self.centralwidget)
        self.predict.setGeometry(QtCore.QRect(360, 290, 111, 51))
        self.predict.setObjectName("predict")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(456, 12, 141, 131))
        self.confidence = QtWidgets.QLabel(self.centralwidget)
        self.confidence.setGeometry(QtCore.QRect(16, 222, 210, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.confidence.setFont(font)
        self.confidence.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.confidence.setObjectName("confidence")
        font = QtGui.QFont()
        font.setFamily("Nirmala UI")
        font.setPointSize(72)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 612, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.main_window = MainWindow

        self.selectFile.clicked.connect(lambda: self.file(MainWindow))
        self.predict.clicked.connect(self.predict_num)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.descText.setText(_translate("MainWindow", "This is an image of : "))
        self.selectFile.setText(_translate("MainWindow", "Select Image"))
        self.predict.setText(_translate("MainWindow", "Predict"))
        self.label.setText(_translate("MainWindow", "?"))
        self.confidence.setText(_translate("MainWindow", "Prediction Confidence: ?"))

    def setImage(self, img):
        """Used to set the the input image"""
        self.img.setPixmap(QtGui.QPixmap(img))

    def setChar(self, char):
        """Used to set the output character"""
        self.label.setText(str(char))

    def file(self, win):
        """Gets the input filename and sets the input image to that image"""
        name = QtWidgets.QFileDialog.getOpenFileName(win, 'Open file')
        self.file_name = name[0]
        self.setImage(name[0])

    def predict_num(self):
        """This method predicts what digit the image previously retrieved is"""
        if self.file_name:
            img_array = self.read_img(self.file_name)
            # plt.imshow(de_noise(img_array), cmap='gray')

            # isDigit = model2.predict(np.array([tf.cast(img_array, tf.float32)]))[0]
            # print(isDigit)
            isDigit = 1
            if isDigit > 0.5:
                prediction_matrix = model.predict(np.array([tf.cast(img_array, tf.float32)]))[0]

                max_val = np.max(prediction_matrix)
                print(max_val, np.argmax(prediction_matrix))
                if max_val > 0.5:
                    prediction = np.argmax(prediction_matrix)
                    self.confidence.setText(f'Prediction Confidence: {str(round(max_val*100, 2))}')
                else:
                    prediction = -1
                    self.confidence.setText(f'Prediction Confidence: ?')

                self.setChar(prediction)
            plt.show()


    def read_img(self, path):
        """This method reads the image whose path was previously retrieved and preprocess it"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(img, (28, 28))
        inverted = np.invert(np.array(resized).reshape((28, 28, 1)))
        denoised_and_reshaped = de_noise(inverted).reshape((28, 28, 1))
        normalized = denoised_and_reshaped / 255.
        return normalized


def create_model():
    """This was the model used in the above defined class"""
    model = tf.keras.Sequential([
        Conv2D(256, 5, input_shape=[28, 28, 1]),
        BatchNormalization(),
        Activation('relu'),
        
        Conv2D(256, 5),
        BatchNormalization(),
        Activation('relu'),
        
        Conv2D(256, 5),
        MaxPooling2D(2),
        BatchNormalization(),
        Activation('relu'),
        
        Flatten(),
        Dense(256),
        Dropout(.2),
        
        Dense(10),
        Activation('softmax')
    ])
    
    return model

def de_noise(img):
    img = img.reshape((784,))
    pct1 = 0.8*255
    pct2 = 0.6*255
    new = []
    for x in img:
        if x >= pct1:
            new.append(255)
        elif x <= pct2:
            new.append(0)
        else:
            new.append(x)

    return np.array(new).reshape((28, 28))



if __name__ == '__main__':
    model = create_model()
    # quit()
    model.load_weights("digit_recog_model/ckpt_03")  # Load the weights of the pretrained model
    # model2 = create_model2()
    # model2.load_weights('convnets/mdl3')
    
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys


class AppWindow(QMainWindow):
    def __init__(self):
        super(AppWindow, self).__init__()
        self.x_pos, self.y_pos = 200, 200
        self.width, self.height = 300, 300
        title = 'QPI Analysis'
        self.setGeometry(self.x_pos, self.y_pos, self.width, self.height)
        self.setWindowTitle(title)
        self.initUI()

    def initUI(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setText('Upload a file:')
        self.label.move(100, 100)

        self.btn1 = QtWidgets.QPushButton(self)
        self.btn1.setText('Browse...')
        self.btn1.move(100, 150)
        self.btn1.clicked.connect(self.onclick)

    def onclick(self):
        if self.label.text() == 'Browse...':
            self.label.setText('Hi!')
        else:
            self.label.setText('Browse...')
        self.update()

    def update(self):
        self.label.adjustSize()


def window():
    app = QApplication(sys.argv)
    win = AppWindow()
    win.show()
    sys.exit(app.exec_())


def boop():
    print('Boop!')


window()

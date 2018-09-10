import sys
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QBrush


class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(200, 200, 800, 400)
        self.setWindowTitle('Colours')
        self.show()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawRectangles(qp)
        qp.end()

    def drawRectangles(self, qp):
        # 長方形の周りの線の色を指定（RGB）
        col = QColor(0, 0, 0)
        # 長方形の周りの線の色を指定（16進数）
        col.setNamedColor('#d4d4d4')
        # ペンの色をセット
        qp.setPen(col)

        # Qcolor(Red, Green, Blue, Alpha) Alphaは透明度
        # drawRect(x, y, width, height)
        qp.setBrush(QColor(255, 80, 0, 160))
        qp.drawRect(130, 15, 90, 60)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

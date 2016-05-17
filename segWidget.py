#!/usr/bin/env python


from PyQt4 import QtCore, QtGui
import sys
from deal_with_tif import load_crop
import numpy as np
import scipy.misc as sci

import forest_testing
import forest_training


class segViewer(QtGui.QMainWindow):
#class segViewer(QtGui.QWidget):
    def __init__(self):
        super(segViewer, self).__init__()

        self.printer = QtGui.QPrinter()
        self.scaleFactor = 0.0
        self.x0 = 10
        self.y0 = 10
        self.x1 = 100
        self.y1 = 100

        self.rubX0 = 0
        self.rubY0 = 0
        self.rubX1 = 0
        self.rubY1 = 0

        self.defRawPath = "./data/rawData.tif"
        self.defGroundPath = "./data/groundTruth.tif"
        self.toSegment = "./data/rawData.tif"

        if len(sys.argv) > 1:
            self.toSegment = sys.argv[1]
            print self.toSegment


        self.rawList = []
        self.GTList = []

    #
    # def paintEvent(self, mousePressEvent):
    #
    #     qp = QtGui.QPainter()
    #     qp.begin(self)
    #     qp.drawRect(10, 15, 90, 60)
    #     qp.end()
    #


        # self.g = QtGui.QGraphicsView()
        # self.g.setGeometry(0,0,1000,790)
        # self.g.scene = QtGui.QGraphicsScene()
        # self.g.setScene(self.g.scene)
        # self.g.item = QtGui.QGraphicsEllipseItem(self.x0, self.y0, self.x1, self.y1)
        # self.g.scene.ad

        #####
        self.rubberband = QtGui.QRubberBand(QtGui.QRubberBand.Rectangle, self)



        # Setting a layout for proper display
        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)

        # Loading the image
        self.imageLabel = QtGui.QLabel()
        self.imageLabel.setBackgroundRole(QtGui.QPalette.Base)
        self.imageLabel.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        # Creating the "Add" button
        btn = QtGui.QPushButton("Add", self)
        btn.clicked.connect(self.add)
        btn.move(1010, 10)

        # Creating the "Segment" button
        btn = QtGui.QPushButton("Segment", self)
        btn.clicked.connect(self.segment)
        btn.move(1010, 30)

        # Creating the scroll area
        self.scrollArea = QtGui.QScrollArea()
        self.scrollArea.setBackgroundRole(QtGui.QPalette.Dark)
        self.scrollArea.setGeometry(0,0,1000,790)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)
        hbox.addWidget(self.scrollArea)
        #hbox.addWidget(btn)
        self.createActions()
        self.createMenus()

        self.setLayout(hbox)
        self.setWindowTitle("Segmentation Viewer")
        self.resize(1150, 800)

        # Read the image path from the command line, otherwise use default
        if (len(sys.argv) > 1):
            self.open(sys.argv[1])
        else:
            self.open()

    #
    # def paintEvent(self, event):
    #
    #     painter = QtGui.QPainter()
    #     painter.begin(self)
    #     painter.fillRect(event.rect(), QtGui.QBrush(QtCore.Qt.white))
    #     painter.setRenderHint(QtGui.QPainter.Antialiasing)
    #     painter.setPen(QtGui.QPen(QtGui.QBrush(QtCore.Qt.red), 1, QtCore.Qt.DashLine))
    #     painter.drawRect(self.largest_rect)
    #     painter.setPen(QtGui.QPen(QtCore.Qt.black))
    #     painter.drawRect(self.clip_rect)
    #     for i in range(4):
    #         painter.drawRect(self.corner(i))
    #
    #     painter.setClipRect(self.clip_rect)
    #     painter.drawPolyline(self.polygon)
    #     painter.setBrush(QtGui.QBrush(QtCore.Qt.blue))
    #     painter.drawPath(self.path)
    #     painter.end()
    #######


    def mousePressEvent(self, event):

        self.rubberband.hide()
        self.coordInitRub(event.pos().x(), event.pos().y())
        self.coordFinRub(event.pos().x(), event.pos().y())
        self.coordInit(event.pos().x() + self.scrollArea.horizontalScrollBar().value(),
                       event.pos().y() + self.scrollArea.verticalScrollBar().value())

        self.coordInitRub(event.pos().x(), event.pos().y())

        self.rubberband.setGeometry(self.rubX0, self.rubY0, self.rubX1 - self.rubX0, self.rubY1 - self.rubY0)


        self.rubberband.show()
        print "init (" + str(self.x0) + ", " + str(self.y0) + ")"

    def mouseMoveEvent(self, event):

        #self.rubberband.setGeometry(self.x0, self.y0, event.pos().x() - self.x0, event.pos().y() - self.y0)
        self.coordFinRub(event.pos().x(), event.pos().y())

        self.rubberband.setGeometry(self.rubX0, self.rubY0, self.rubX1 - self.rubX0, self.rubY1 - self.rubY0)

        self.rubberband.show()


    def mouseReleaseEvent(self, QMouseEvent):
        self.coordFin(QMouseEvent.pos().x() + self.scrollArea.horizontalScrollBar().value(),
                       QMouseEvent.pos().y() + self.scrollArea.verticalScrollBar().value())
        print "final (" + str(self.x1) + ", " + str(self.y1) + ")"

################################################################
    def add(self):
        print "hello"

        cropRaw = load_crop(image_path = self.defRawPath, origin = (self.y0, self.x0), length=(self.y1 - self.y0), width=(self.x1 - self.x0))
        cropGT = load_crop(image_path = self.defGroundPath, origin = (self.y0, self.x0), length=(self.y1 - self.y0), width=(self.x1 - self.x0))
        self.rawList.append(cropRaw)
        self.GTList.append(cropGT)
       # sci.imsave("rawTest.png", cropRaw, "png")
       # sci.imsave("rawGT.png   ", cropGT, "png")

    def segment(self):
        print "segment"

        #execfile('./forest_training.py')
        #execfile('./forest_testing.py')
        forest_training.trainingExec(self.rawList, self.GTList)
        forest_testing.classifyExec(path=self.toSegment)


################################################################

    def coordInit(self, x = 0, y = 0):
        self.x0 = x
        self.y0 = y
    def coordFin(self, x = 0, y = 0):
        self.x1 = x
        self.y1 = y

    def coordInitRub(self, x = 0, y = 0):
        self.rubX0 = x
        self.rubY0 = y
    def coordFinRub(self, x = 0, y = 0):
        self.rubX1 = x
        self.rubY1 = y


    def open(self, fileName = None):

        path  = self.defRawPath
        fileName = QtGui.QImage(path)
        image = QtGui.QImage(fileName)
        if image.isNull():
            QtGui.QMessageBox.information(self, "Image Viewer",
                    "Cannot load %s." % fileName)
            return

        self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(image))
        self.scaleFactor = 1.0

        self.printAct.setEnabled(True)
        self.fitToWindowAct.setEnabled(True)
        self.updateActions()

        if not self.fitToWindowAct.isChecked():
            self.imageLabel.adjustSize()

    def print_(self):
        dialog = QtGui.QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QtGui.QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), QtCore.Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QtGui.QMessageBox.about(self, "About Image Viewer",
                "<p>The <b>Image Viewer</b> example shows how to combine "
                "QLabel and QScrollArea to display an image. QLabel is "
                "typically used for displaying text, but it can also display "
                "an image. QScrollArea provides a scrolling view around "
                "another widget. If the child widget exceeds the size of the "
                "frame, QScrollArea automatically provides scroll bars.</p>"
                "<p>The example demonstrates how QLabel's ability to scale "
                "its contents (QLabel.scaledContents), and QScrollArea's "
                "ability to automatically resize its contents "
                "(QScrollArea.widgetResizable), can be used to implement "
                "zooming and scaling features.</p>"
                "<p>In addition the example shows how to use QPainter to "
                "print an image.</p>")

    def createActions(self):
        self.openAct = QtGui.QAction("&Open...", self, shortcut="Ctrl+O",
                triggered=self.open)

        self.printAct = QtGui.QAction("&Print...", self, shortcut="Ctrl+P",
                enabled=False, triggered=self.print_)

        self.exitAct = QtGui.QAction("E&xit", self, shortcut="Ctrl+Q",
                triggered=self.close)

        self.zoomInAct = QtGui.QAction("Zoom &In (25%)", self,
                shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)

        self.zoomOutAct = QtGui.QAction("Zoom &Out (25%)", self,
                shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)

        self.normalSizeAct = QtGui.QAction("&Normal Size", self,
                shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)

        self.fitToWindowAct = QtGui.QAction("&Fit to Window", self,
                enabled=False, checkable=True, shortcut="Ctrl+F",
                triggered=self.fitToWindow)

        self.aboutAct = QtGui.QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QtGui.QAction("About &Qt", self,
                triggered=QtGui.qApp.aboutQt)

    def createMenus(self):
        self.fileMenu = QtGui.QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QtGui.QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QtGui.QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                                + ((factor - 1) * scrollBar.pageStep()/2)))


if __name__ == '__main__':

    import sys

    app = QtGui.QApplication(sys.argv)
    segViewer = segViewer()
    segViewer.show()
    sys.exit(app.exec_())

#---------------------------------------------------------------
# Copyright (C) 2021 Bone Imaging Laboratory
# All rights reserved.
# bonelab@ucalgary.ca
#---------------------------------------------------------------
# Created February 2, 2021
# Steven Boyd
#---------------------------------------------------------------
# The MainWindow class is defined for the Qt GUI as well as
# the signals and slots that make the GUI functional.
#---------------------------------------------------------------

import os
import sys
import pickle
import numpy as np
import vtk
import matplotlib.pyplot as plt
from scipy import ndimage

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg

class MainWindow(qtw.QMainWindow):

  def __init__(self,
               input_file,
               gaussian,
               radius,
               thresh,
               zoom,
               zSlice,
               brightness,
               window_size,
               colorWindow,
               colorLevel,
               *args, **kwargs):
    """MainWindow constructor"""
    super().__init__(*args, **kwargs)

    # Window setup
    self.resize(window_size[0],window_size[1])
    self.title = "Qt Viewer for Lesion Augmentation"

    self.statusBar().showMessage("Welcome.",8000)

    # Capture defaults
    self.gaussian = gaussian
    self.radius = radius
    self.thresh = thresh
    self.zoom = zoom
    self.brightness = brightness
    self.shape_dic = None
    self.lesion_dic = {}
    self.thresholdArray = None
    self.imageArray = None
    self.zSlice = 100
    self.shape = None
    self.crop = None
    self.colorWindow = colorWindow
    self.colorLevel = colorLevel
    # Initialize the window
    self.initUI()

    # Set up some VTK pipeline classes
    self.reader = None
    self.gauss = vtk.vtkImageGaussianSmooth()
    self.lesion = vtk.vtkImageData()
    self.threshold = vtk.vtkImageThreshold()
    self.mapToColors = vtk.vtkImageMapToColors()
    self.coordinate = vtk.vtkCoordinate()
    self.imageViewer = vtk.vtkImageViewer2()

    self.resizeImage = vtk.vtkImageResize()
    self.resizeSeg = vtk.vtkImageResize()

    self.contourRep = vtk.vtkOrientedGlyphContourRepresentation()
    self.contourWidget = vtk.vtkContourWidget()
    self.placer = vtk.vtkImageActorPointPlacer()

    self.polyData = None

    self.origmapper = vtk.vtkImageMapper()#vtkImageSliceMapper()#
    self.mapper = vtk.vtkImageMapper()
    self.stencilmapper = vtk.vtkPolyDataMapper()

    self.origactor = vtk.vtkActor2D() #vtkImageActor()
    self.actor = vtk.vtkActor2D()
    self.stencilactor = vtk.vtkActor()
    # Take inputs from command line. Only use these if there is an input file specified
    if (input_file != None):
      if (not os.path.exists(input_file)):
        qtw.QMessageBox.warning(self, "Error", "Invalid input file.")
        return

      self.createPipeline(input_file)
      self.statusBar().showMessage("Loading file " + input_file,4000)
      self.changeSigma(gaussian)
      self.changeRadius(radius)
      self.changeThreshold(thresh)
      self.changeBrightness(brightness)
      self.changeSlice(zSlice)
      self.changeLevel(5000)
      self.changeWindow(5000)

  def initUI(self):
    ########################################
    # Create Widgets
    ########################################

    self.loadPushButton = qtw.QPushButton(
      "Load Image",
      self,
      objectName = "loadPushButton",
      shortcut=qtg.QKeySequence("Ctrl+f")
    )
    self.sigmaSpinBox = qtw.QDoubleSpinBox(
      self,
      objectName = "sigmaSpinBox",
      value=self.gaussian,
      decimals=1,
      maximum=20.0,
      minimum=0.1,
      singleStep=0.1,
      keyboardTracking=False
    )
    self.radiusSpinBox = qtw.QSpinBox(
      self,
      objectName = "radiusSpinBox",
      value=self.radius,
      maximum=20,
      minimum=1,
      singleStep=1,
      keyboardTracking=False
    )

    self.threshSpinBox = qtw.QSpinBox(
      self,
      objectName = "threshSpinBox",
      value=self.thresh,
      maximum=3000,
      minimum=-3000,
      singleStep=5,
      keyboardTracking=False
    )

    self.brightnessSpinBox = qtw.QSpinBox(
      self,
      objectName = "brightnessSpinBox",
      value=self.brightness,
      maximum=3000,
      minimum=-3000,
      singleStep=5,
      keyboardTracking=False
    )

    self.sliceSpinBox = qtw.QSpinBox(
      self,
      objectName = "sliceSpinBox",
      value=self.zSlice,
      maximum=3000,
      minimum=-3000,
      singleStep=1,
      keyboardTracking=False
    )

    self.levelSpinBox = qtw.QSpinBox(
      self,
      objectName = "levelSpinBox",
      value=5000,
      maximum=5000,
      minimum=-3000,
      singleStep=50,
      keyboardTracking=False
    )

    self.windowSpinBox = qtw.QSpinBox(
      self,
      objectName = "windowSpinBox",
      value=self.colorWindow,
      maximum=5000,
      minimum=-3000,
      singleStep=50,
      keyboardTracking=False
    )

    self.lesionPushButton = qtw.QPushButton(
      "Add Lesion",
      self,
      objectName = "lesionPushButton",
      shortcut=qtg.QKeySequence("Ctrl+l")
    )

    self.savePushButton = qtw.QPushButton(
      "Save Lesion",
      self,
      objectName = "savePushButton",
      shortcut=qtg.QKeySequence("Ctrl+l")
    )

    self.deletePushButton = qtw.QPushButton(
      "Delete Lesion",
      self,
      objectName = "deletePushButton",
      shortcut=qtg.QKeySequence("Ctrl+l")
    )

    self.resetPushButton = qtw.QPushButton(
      "Reset Contour",
      self,
      objectName = "resetPushButton",
      shortcut=qtg.QKeySequence("Ctrl+l")
    )

    self.saveCropPushButton = qtw.QPushButton(
      "Save Cropped Region",
      self,
      objectName = "saveCropPushButton",
      shortcut=qtg.QKeySequence("Ctrl+l")
    )

    self.confirmCropPushButton = qtw.QPushButton(
      "Confirm Position of Cropped Contour",
      self,
      objectName = "confirmCropPushButton",
      shortcut=qtg.QKeySequence("Ctrl+l")
    )

    self.addCropPushButton = qtw.QPushButton(
      "Add Cropped Pixels to Current Slice",
      self,
      objectName = "addCropPushButton",
      shortcut=qtg.QKeySequence("Ctrl+l")
    )
    # Create the menu options --------------------------------------------------------------------
    menubar = qtw.QMenuBar()
    self.setMenuBar(menubar)
    menubar.setNativeMenuBar(False)

    file_menu = menubar.addMenu("File")
    open_action = file_menu.addAction("Open Image")
    file_menu.addSeparator()
    about_action = file_menu.addAction("About")
    quit_action = file_menu.addAction("Quit")

    # Lay out the GUI ----------------------------------------------------------------------------
    self.mainGroupBox = qtw.QGroupBox("Image Controls")
    self.mainGroupBox.setLayout(qtw.QGridLayout())#QHBoxLayout())

    self.controlsGroupBox = qtw.QGroupBox("Thresholding controls")
    self.controlsGroupBox.setLayout(qtw.QVBoxLayout())
    self.controlsFormLayout = qtw.QFormLayout()
    self.controlsFormLayout.addRow("Sigma",self.sigmaSpinBox)
    self.controlsFormLayout.addRow("Radius",self.radiusSpinBox)
    self.controlsFormLayout.addRow("Global Threshold",self.threshSpinBox)
    self.controlsFormLayout.addRow("Lesion Brightness",self.brightnessSpinBox)
    self.controlsFormLayout.addRow("Slice Index", self.sliceSpinBox)
    self.controlsFormLayout.addRow("Color Level", self.levelSpinBox)
    self.controlsFormLayout.addRow("Color Window", self.windowSpinBox)
    self.controlsGroupBox.layout().addLayout(self.controlsFormLayout)

    self.mainGroupBox.layout().addWidget(self.loadPushButton)
    self.mainGroupBox.layout().addWidget(self.lesionPushButton)
    self.mainGroupBox.layout().addWidget(self.savePushButton)
    self.mainGroupBox.layout().addWidget(self.deletePushButton)
    self.mainGroupBox.layout().addWidget(self.resetPushButton)
    self.mainGroupBox.layout().addWidget(self.controlsGroupBox)

    self.mainGroupBox.layout().addWidget(self.saveCropPushButton)
    self.mainGroupBox.layout().addWidget(self.confirmCropPushButton)
    self.mainGroupBox.layout().addWidget(self.addCropPushButton)


    # Assemble the side control panel and put it in a QPanel widget ------------------------------
    self.panel = qtw.QVBoxLayout()
    self.panel.addWidget(self.mainGroupBox)
    self.panelWidget = qtw.QFrame()
    self.panelWidget.setLayout(self.panel)

    # Create the VTK rendering window ------------------------------------------------------------
    self.vtkWidget = QVTKRenderWindowInteractor()
    self.vtkWidget.AddObserver("ExitEvent", lambda o, e, a=self: a.quit())
    #self.vtkWidget.AddObserver("MouseWheelForwardEvent", self.wheelForward)
    #self.vtkWidget.AddObserver("MouseWheelBackwardEvent", self.wheelBackward)
    #self.vtkWidget.AddObserver("KeyPressEvent", self.keyPressEvent)

    # Create main layout and add VTK window and control panel
    self.mainLayout = qtw.QHBoxLayout()
    self.mainLayout.addWidget(self.vtkWidget,4)
    self.mainLayout.addWidget(self.panelWidget,1)

    self.frame = qtw.QFrame()
    self.frame.setLayout(self.mainLayout)
    self.setCentralWidget(self.frame)

    self.setWindowTitle(self.title)
    self.centreWindow()

    # Set size policies --------------------------------------------------------------------------
    self.sigmaSpinBox.setMinimumSize(70,20)
    self.radiusSpinBox.setMinimumSize(70,20)
    self.threshSpinBox.setMinimumSize(70,20)
    self.brightnessSpinBox.setMinimumSize(70,20)
    self.sliceSpinBox.setMinimumSize(70,20)
    self.levelSpinBox.setMinimumSize(70,20)
    self.windowSpinBox.setMinimumSize(70,20)

    self.mainGroupBox.setMaximumSize(1000,1000)

    self.vtkWidget.setSizePolicy(
      qtw.QSizePolicy.MinimumExpanding,
      qtw.QSizePolicy.MinimumExpanding
    )

    self.mainGroupBox.setSizePolicy(
      qtw.QSizePolicy.Maximum,
      qtw.QSizePolicy.Maximum
    )

    # Connect signals and slots ------------------------------------------------------------------
    self.loadPushButton.clicked.connect(self.openFile)
    self.sigmaSpinBox.valueChanged.connect(lambda s: self.changeSigma(s))
    self.radiusSpinBox.valueChanged.connect(lambda s: self.changeRadius(s))
    self.threshSpinBox.valueChanged.connect(lambda s: self.changeThreshold(s))
    self.brightnessSpinBox.valueChanged.connect(lambda s: self.changeBrightness(s))
    self.sliceSpinBox.valueChanged.connect(lambda s: self.changeSlice(s))
    self.windowSpinBox.valueChanged.connect(lambda s: self.changeWindow(s))
    self.levelSpinBox.valueChanged.connect(lambda s: self.changeLevel(s))

    self.lesionPushButton.clicked.connect(self.addLesion)
    self.savePushButton.clicked.connect(self.saveLesion)
    self.deletePushButton.clicked.connect(self.deleteLesion)
    self.resetPushButton.clicked.connect(self.resetContour)

    self.saveCropPushButton.clicked.connect(self.saveCrop)
    self.confirmCropPushButton.clicked.connect(self.confirmCrop)
    self.addCropPushButton.clicked.connect(self.addCrop)
    self.initRenderWindow()

    # Menu actions
    open_action.triggered.connect(self.openFile)
    about_action.triggered.connect(self.about)
    quit_action.triggered.connect(self.quit)

    self.pipe = None

    # End main UI code
    self.show()

    ########################################
    # Define methods for controlling GUI
    ########################################

  def centreWindow(self):
    qr = self.frameGeometry()
    cp = qtw.QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    self.move(qr.topLeft())

  def initRenderWindow(self):
    # Create renderer
    self.renderer = vtk.vtkRenderer()
    self.renderer.SetBackground((.2, .2, .2)) # grey
    #self.renderer.WorldToView()
    # Create interactor
    self.renWin = self.vtkWidget.GetRenderWindow()
    self.renWin.AddRenderer(self.renderer)
    self.iren = self.renWin.GetInteractor()
    self.iren.RemoveObservers("LeftButtonPressEvent")
    #self.iren.SetInteractorStyle(vtk.vtkInteractorStyleImage().SetInteractionModeToImage3D())

    #self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTerrain())
    #self.iren.SetInteractorStyle(vtk.vtkInteractionModeToImage)
    # Initialize
    self.iren.Initialize()
    self.iren.Start()


  def refreshRenderWindow(self):
    self.renWin.Render()
    self.iren.Render()

  def createPipeline(self, _filename):
    # Read in the file
    if _filename.lower().endswith('.nii'):
      self.reader = vtk.vtkNIFTIImageReader()
      self.reader.SetFileName(_filename)
    elif _filename.lower().endswith('.nii.gz'):
      self.reader = vtk.vtkNIFTIImageReader()
      self.reader.SetFileName(_filename)
    elif _filename.lower().endswith('.dcm'):
      self.reader = vtk.vtkDICOMImageReader()
      self.reader.SetDirectoryName(os.path.dirname(_filename))
    elif os.path.isdir(_filename):
      self.reader = vtk.vtkDICOMImageReader()
      self.reader.SetDirectoryName(_filename)

    if self.reader is None:
        os.sys.exit("[ERROR] Cannot find reader for file \"{}\"".format(self.filename))
    self.reader.Update()

    #global threshold
    self.threshold.SetInputConnection(self.reader.GetOutputPort())
    self.threshold.ThresholdByUpper(self.thresh)
    self.threshold.SetInValue(1)
    self.threshold.SetOutValue(0)
    self.threshold.Update()

    #self.read_dictionary()
    self.convertImageData()
    self.convertThresholdData()
    self.initializeLesion()

    #Gaussian smoothing
    self.gauss.SetStandardDeviation(self.gaussian, self.gaussian, self.gaussian)
    self.gauss.SetRadiusFactors(self.radius, self.radius, self.radius)
    self.gauss.SetInputData(self.lesion)

    lookupTable = vtk.vtkLookupTable()
    lookupTable.SetNumberOfTableValues(2)
    lookupTable.SetRange(0.0,1.0)
    lookupTable.SetTableValue( 0, 0.0, 0.0, 0.0, 0.0 ) #label outRangeValue is transparent
    lookupTable.SetTableValue( 1, 0.0, 1.0, 0.0, 1.0 )  #label inRangeValue is opaque and green
    lookupTable.Build()

    self.mapToColors.SetLookupTable(lookupTable)
    self.mapToColors.PassAlphaToOutputOn()
    self.mapToColors.SetInputConnection(self.threshold.GetOutputPort())

    #Resize segmented image
    self.resizeSeg.SetInputConnection(self.mapToColors.GetOutputPort())
    self.resizeSeg.SetResizeMethodToMagnificationFactors()
    self.resizeSeg.SetMagnificationFactors(self.zoom,self.zoom,0)

    # Resize original image
    self.resizeImage.SetInputConnection(self.gauss.GetOutputPort())
    self.resizeImage.SetResizeMethodToMagnificationFactors()
    self.resizeImage.SetMagnificationFactors(self.zoom,self.zoom,0)

    #
    self.imageViewer.SetInputConnection(self.resizeImage.GetOutputPort())
    #self.imageViewer.SetSliceOrientationToXZ()
    self.imageViewer.SetRenderWindow(self.renWin)
    self.imageViewer.SetSlice(self.zSlice)
    self.imageViewer.Render()

    # pklace contour on plane aligned with image actor
    self.placer.SetImageActor(self.imageViewer.GetImageActor())
    #self.placer.SetBounds(-256,256,-256,256,0,0)
    #self.contourRep.SetPixelTolerance(1)
    self.contourRep.SetPointPlacer(self.placer)
    self.contourRep.GetProperty().SetColor(0,1,0)
    self.contourRep.GetLinesProperty().SetColor(0,1,0)




    #contour widget
    self.contourWidget.SetRepresentation(self.contourRep)
    self.contourWidget.SetInteractor(self.iren)
    self.contourWidget.FollowCursorOn()
    self.contourWidget.SetContinuousDraw(1)
    self.contourWidget.SetEnabled(True)
    self.contourWidget.ProcessEventsOn()
    self.contourWidget.CloseLoop()

    self.refreshRenderWindow()

    return

  def resetContour(self):
    #self.contourRep.ClearAllNodes()
    #self.contourWidget.ResetAction()
    self.contourWidget.Initialize()
    self.refreshRenderWindow()
    return


  def convertImageData(self):

    # convert vtk image data to numpy array
    self.reader.Update()
    imageData = self.reader.GetOutput()
    imageDataArray = imageData.GetPointData().GetScalars()
    imageArray = vtk_to_numpy(imageDataArray).reshape(imageData.GetDimensions(), order='F')
    self.imageArray = imageArray
    return

  def convertThresholdData(self):
    # convert thresholding mask for analysis
    self.threshold.Update()
    thresholdData = self.threshold.GetOutput()
    thresholdDataArray = thresholdData.GetPointData().GetScalars()
    thresholdArray = vtk_to_numpy(thresholdDataArray).reshape(thresholdData.GetDimensions(), order='F')
    self.thresholdArray = thresholdArray

    return

  def initializeLesion(self):
    self.reader.Update()
    self.lesion.CopyStructure(self.reader.GetOutput())
    self.lesion.GetPointData().SetScalars(numpy_to_vtk(num_array=
                                    self.imageArray.ravel(order='F'),
                                    deep=True))

  def saveCrop(self):
    global nodes
    state = self.contourWidget.GetWidgetState()
    dim = self.reader.GetOutput().GetDimensions()
    self.zSlice = self.imageViewer.GetSlice()
    if state == 0:
        self.statusBar().showMessage(f"Draw lesion before saving",4000)
    elif state == 2:
        self.polyData = self.contourRep.GetContourRepresentationAsPolyData()
        nodes = vtk_to_numpy(self.polyData.GetPoints().GetData())
        mask = np.zeros([dim[0],dim[1]])
        contour = np.floor(nodes).astype(int)
        mask[contour[:,0], contour[:,1]] = 1
        binary  = ndimage.morphology.binary_fill_holes(mask)
        self.shape = binary
        plt.figure()
        plt.imshow(binary)
        plt.gca().invert_yaxis()
        plt.show()

        crop = self.imageArray[:,:,self.zSlice][binary]
        self.crop = crop
    return

  def confirmCrop(self):
    global nodes
    state = self.contourWidget.GetWidgetState()
    dim = self.reader.GetOutput().GetDimensions()
    if state == 0:
        self.statusBar().showMessage(f"Draw lesion before saving",4000)
    elif state == 2:
        self.polyData = self.contourRep.GetContourRepresentationAsPolyData()
        nodes = vtk_to_numpy(self.polyData.GetPoints().GetData())
        mask = np.zeros([dim[0],dim[1]])
        contour = np.floor(nodes).astype(int)
        mask[contour[:,0], contour[:,1]] = 1
        binary  = ndimage.morphology.binary_fill_holes(mask)
        self.shape = binary
        plt.figure()
        plt.imshow(binary)
        plt.gca().invert_yaxis()
        plt.show()

  def addCrop(self):
    state = self.contourWidget.GetWidgetState()
    if state == 0:
        self.statusBar().showMessage(f"Draw and save lesion before adding to slice",4000)
    else:
        #add lesion to current zSlice
        self.zSlice = self.imageViewer.GetSlice()
        lesionDataArray = self.lesion.GetPointData().GetScalars()
        lesionArray = vtk_to_numpy(lesionDataArray).reshape(self.lesion.GetDimensions(), order='F')


        img_slice = lesionArray[:,:,self.zSlice]
        if self.shape.shape[0] != img_slice.shape[0]:
            self.shape = np.transpose(self.shape)

        idx = min([img_slice[self.shape].shape, self.crop.shape])[0]
        #normalize cropped value
        MAX = img_slice.max()
        MIN = img_slice.min()
        #img_slice[self.shape][:idx] = self.crop[:idx]
        if img_slice[self.shape].shape < self.crop.shape:
            img_slice[self.shape]= self.crop[:idx]#* (MAX - MIN) / MAX
        else:
            img_slice[self.shape][:idx] = self.crop
        self.lesion.CopyStructure(self.reader.GetOutput())
        self.lesion.GetPointData().SetScalars(numpy_to_vtk(num_array=
                                        lesionArray.ravel(order='F'),
                                        deep=True))
        #self.lesion_dic[self.zSlice] = self.shape
    self.refreshRenderWindow()


  def saveLesion(self):
    #camera = self.renderer.GetActiveCamera()
    #windowHeight = self.renderer.GetRenderWindow().GetViews()[0].getSize()[1]
    #worldHeight = 2*camera.getParallelScale()
    #if windowHeight > 0:
#         scale = worlHeight / windowHeight

    global nodes


    state = self.contourWidget.GetWidgetState()

    dim = self.reader.GetOutput().GetDimensions()

    if state == 0:
        self.statusBar().showMessage(f"Draw lesion before saving",4000)
    elif state == 2:

        #self.contourRep.SetWorldTolerance(0.00001)
        self.polyData = self.contourRep.GetContourRepresentationAsPolyData()
        nodes = 2.15 * vtk_to_numpy(self.polyData.GetPoints().GetData())


        mask = np.zeros([dim[0],dim[1]])
        contour = np.floor(nodes).astype(int)
        mask[contour[:,0], contour[:,1]] = 1
        binary = ndimage.morphology.binary_fill_holes(mask)

        plt.figure()
        plt.imshow(binary)
        plt.show()
        self.shape = binary
        self.statusBar().showMessage(f"Lesion saved to dictionary",4000)
    return

  def deleteLesion(self):
    self.zSlice = self.imageViewer.GetSlice()
    if self.zSlice in self.lesion_dic:
        self.threshold.Update()

        lesionDataArray = self.lesion.GetPointData().GetScalars()
        lesionArray = vtk_to_numpy(lesionDataArray).reshape(self.lesion.GetDimensions(), order='F')
        lesionArray[:,:,self.zSlice] = self.imageArray[:,:,self.zSlice]
        self.lesion.GetPointData().SetScalars(numpy_to_vtk(num_array=lesionArray.ravel(order='F'),
                                    deep=True))
    self.refreshRenderWindow()

    return


  def addLesion(self):
    state = self.contourWidget.GetWidgetState()
    if state == 0:
        self.statusBar().showMessage(f"Draw and save lesion before adding to slice",4000)
    else:
        #add lesion to current zSlice
        self.zSlice = self.imageViewer.GetSlice()
        lesionDataArray = self.lesion.GetPointData().GetScalars()
        lesionArray = vtk_to_numpy(lesionDataArray).reshape(self.lesion.GetDimensions(), order='F')

        img_slice = lesionArray[:,:,self.zSlice]
        #if self.shape.shape[0] != img_slice.shape[0]:
    #        self.shape = np.transpose(self.shape)

        img_slice[self.shape] = self.brightness
        print(self.brightness)
        plt.figure()
        plt.imshow(img_slice, cmap='gray')
        plt.show()
        self.lesion.CopyStructure(self.reader.GetOutput())
        self.lesion.GetPointData().SetScalars(numpy_to_vtk(num_array=
                                        lesionArray.ravel(order='F'),
                                        deep=True))
        self.lesion_dic[self.zSlice] = self.shape
    self.refreshRenderWindow()



  def addLesion1(self):
    state = self.contourWidget.GetWidgetState()
    if state == 0:
        self.statusBar().showMessage(f"Draw and save lesion before adding to slice",4000)
    else:
        #add lesion to current zSlice
        self.zSlice = self.imageViewer.GetSlice()
        lesionDataArray = self.lesion.GetPointData().GetScalars()
        lesionArray = vtk_to_numpy(lesionDataArray).reshape(self.lesion.GetDimensions(), order='F')


        img_slice = lesionArray[:,:,self.zSlice]
        if self.shape.shape[0] != img_slice.shape[0]:
            self.shape = np.transpose(self.shape)

        img_slice[self.shape] = self.brightness

        self.lesion.CopyStructure(self.reader.GetOutput())
        self.lesion.GetPointData().SetScalars(numpy_to_vtk(num_array=
                                        lesionArray.ravel(order='F'),
                                        deep=True))
        self.lesion_dic[self.zSlice] = self.shape
    self.refreshRenderWindow()




  def adjustBrightness(self):

    self.zSlice = self.imageViewer.GetSlice()
    lesionDataArray = self.lesion.GetPointData().GetScalars()
    lesionArray = vtk_to_numpy(lesionDataArray).reshape(self.lesion.GetDimensions(), order='F')


    if self.zSlice in self.lesion_dic:

        self.shape = self.lesion_dic[self.zSlice]
        #x1,x2,y1,y2 = lesionParams[-4:]
        #image_cut = lesionParams[0]
        #random_shape = lesionParams[1]
        #imageData = self.lesion
        #imageDataArray = imageData.GetPointData().GetScalars()
        #imageArray = vtk_to_numpy(imageDataArray).reshape(imageData.GetDimensions(), order='F')
        img_slice = lesionArray[:,:,self.zSlice]

        img_slice[self.shape] = self.brightness

        #imageArray[x1:x2,y1:y2,zSlice] = image_cut

        self.lesion.GetPointData().SetScalars(numpy_to_vtk(num_array=lesionArray.ravel(order='F'),
                                    deep=True))

    self.refreshRenderWindow()
    return

  def changeSigma(self, _value):
    self.gauss.SetStandardDeviation(_value, _value, _value)
    self.statusBar().showMessage(f"Changing standard deviation to {_value}",4000)
    self.refreshRenderWindow()
    return

  def changeRadius(self, _value):
    self.gauss.SetRadiusFactors(_value, _value, _value)
    self.statusBar().showMessage(f"Changing radius to {_value}",4000)
    self.refreshRenderWindow()
    return

  def changeThreshold(self, _value):
    self.threshold.ThresholdByUpper(_value)
    self.statusBar().showMessage(f"Changing threshold to {_value}",4000)
    self.refreshRenderWindow()
    return

  def changeBrightness(self, _value):
    self.brightness = _value
    self.adjustBrightness()
    self.statusBar().showMessage(f"Changing lesion brightness to {_value}",4000)
    self.refreshRenderWindow()

    return
  def changeSlice(self, _value):
    self.zSlice = _value
    self.imageViewer.SetSlice(self.zSlice)

    self.statusBar().showMessage(f"Changing zSlice to {_value}",4000)
    self.refreshRenderWindow()

    return
  def changeLevel(self, _value):
    self.colorLevel = _value
    self.imageViewer.SetColorLevel(self.colorLevel)
    self.refreshRenderWindow()
    return

  def changeWindow(self, _value):
    self.colorWindow = _value
    self.imageViewer.SetColorWindow(self.colorWindow)
    self.refreshRenderWindow()
    return

  def validExtension(self, extension):
    if (extension == ".nii" or \
        extension == ".dcm" or \
        extension == ".gz"):
      return True
    else:
      return False

  def openFile(self):
    self.statusBar().showMessage("Load image types (.nii, .dcm)",4000)
    filename, _ = qtw.QFileDialog.getOpenFileName(
      self,
      "Select a 3D image file to open…",
      qtc.QDir.homePath(),
      "Nifti Files (*.nii) ;;DICOM Files (*.dcm) ;;All Files (*)",
      "All Files (*)",
      qtw.QFileDialog.DontUseNativeDialog |
      qtw.QFileDialog.DontResolveSymlinks
    )

    if filename:
      _,ext = os.path.splitext(filename)
      if not (self.validExtension(ext.lower())):
        qtw.QMessageBox.warning(self, "Error", "Invalid file type.")
        return

      self.createPipeline(filename)
      self.statusBar().showMessage("Loading file " + filename,4000)
    return

  def quit(self):
    reply = qtw.QMessageBox.question(self, "Message",
      "Are you sure you want to quit?", qtw.QMessageBox.Yes |
      qtw.QMessageBox.No, qtw.QMessageBox.Yes)
    if reply == qtw.QMessageBox.Yes:
      exit(0)

  def about(self):
    about = qtw.QMessageBox(self)
    about.setText("QtBasic 1.0")
    about.setInformativeText("Copyright (C) 2021\nBone Imaging Laboratory\nAll rights reserved.\nbonelab@ucalgary.ca")
    about.setStandardButtons(qtw.QMessageBox.Ok | qtw.QMessageBox.Cancel)
    about.exec_()
    #idx = np.argwhere(self.thresholdArray[:,:,zSlice]==1)

    #random_index = [np.random.randint(0,len(self.shape_dic),1),
    #                np.random.randint(0,len(idx),1)]
    #random_shape = self.shape_dic[random_index[0][0]]

    #random_shape = self.shape
    #print('lesion shape', random_shape.shape)
    #top_right_pos = idx[random_index[1][0]]
    #x1, x2 = top_right_pos[0], top_right_pos[0] + random_shape.shape[0]
    #y1, y2 = top_right_pos[1],top_right_pos[1] + random_shape.shape[1]
    #image_cut = img_copy[x1:x2, y1:y2,zSlice]
    #print('image_cut', image_cut.shape)
    #image_cut[random_shape] = self.brightness

    #self.lesion_dic[zSlice] = [image_cut, random_shape, x1, x2, y1, y2]
    # make vtk image data structure for new image with lesion

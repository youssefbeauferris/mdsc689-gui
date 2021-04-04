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
               *args, **kwargs):
    """MainWindow constructor"""
    super().__init__(*args, **kwargs)

    # Window setup
    self.resize(window_size[0],window_size[1])
    self.title = "Basic Qt Viewer for MDSC 689.03"

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
    # Initialize the window
    self.initUI()

    # Set up some VTK pipeline classes
    self.reader = None
    self.gauss = vtk.vtkImageGaussianSmooth()
    self.lesion = vtk.vtkImageData()
    self.threshold = vtk.vtkImageThreshold()
    self.mapToColors = vtk.vtkImageMapToColors()

    self.imageViewer = vtk.vtkImageViewer2()

    self.resizeImage = vtk.vtkImageResize()
    self.resizeSeg = vtk.vtkImageResize()

    self.contourRep = vtk.vtkOrientedGlyphContourRepresentation()
    self.contourWidget = vtk.vtkContourWidget()
    self.placer = vtk.vtkImageActorPointPlacer()

    self.stencil = vtk.vtkImageStencilToImage()
    self.polydata = vtk.vtkPolyDataToImageStencil()


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
      singleStep=5,
      keyboardTracking=False
    )


    self.lesionPushButton = qtw.QPushButton(
      "Add Lesion",
      self,
      objectName = "lesionPushButton",
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
    self.mainGroupBox.setLayout(qtw.QHBoxLayout())

    self.controlsGroupBox = qtw.QGroupBox("Thresholding controls")
    self.controlsGroupBox.setLayout(qtw.QVBoxLayout())
    self.controlsFormLayout = qtw.QFormLayout()
    self.controlsFormLayout.addRow("Sigma",self.sigmaSpinBox)
    self.controlsFormLayout.addRow("Radius",self.radiusSpinBox)
    self.controlsFormLayout.addRow("Global Threshold",self.threshSpinBox)
    self.controlsFormLayout.addRow("Lesion Brightness",self.brightnessSpinBox)
    self.controlsFormLayout.addRow("Slice Index", self.sliceSpinBox)
    self.controlsGroupBox.layout().addLayout(self.controlsFormLayout)

    self.mainGroupBox.layout().addWidget(self.loadPushButton)
    self.mainGroupBox.layout().addWidget(self.lesionPushButton)
    self.mainGroupBox.layout().addWidget(self.controlsGroupBox)

    # Assemble the side control panel and put it in a QPanel widget ------------------------------
    self.panel = qtw.QVBoxLayout()
    self.panel.addWidget(self.mainGroupBox)
    self.panelWidget = qtw.QFrame()
    self.panelWidget.setLayout(self.panel)

    # Create the VTK rendering window ------------------------------------------------------------
    self.vtkWidget = QVTKRenderWindowInteractor()
    self.vtkWidget.AddObserver("ExitEvent", lambda o, e, a=self: a.quit())
    self.vtkWidget.AddObserver("MouseWheelForwardEvent", self.wheelForward)
    self.vtkWidget.AddObserver("MouseWheelBackwardEvent", self.wheelBackward)
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

    self.mainGroupBox.setMaximumSize(1000,200)

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
    self.lesionPushButton.clicked.connect(self.addLesion)
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

    self.read_dictionary()
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
    self.imageViewer.SetRenderWindow(self.renWin)
    #self.imageViewer.SetInteractor(self.iren)
    self.imageViewer.SetSlice(self.zSlice)

    self.imageViewer.Render()
    #self.imageViewer.GetRenderer().ResetCamera()
    #self.imageViewer.Render()

    self.placer.SetImageActor(self.imageViewer.GetImageActor())
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

    #convert contour widget to poly data so that it can be passed to stencil
    #creating stencil to fill in closed contour

    self.polydata.SetInputData(self.contourRep.GetContourRepresentationAsPolyData())
    #self.polydata.SetOutputOrigin(self.resizeImage.GetOrigin())
    self.polydata.Update()

    self.stencil.SetInputConnection(self.polydata.GetOutputPort())
    self.stencil.SetInsideValue(1)
    self.stencil.Update()

    # Set mapper for image data
    self.origmapper.SetInputConnection(self.resizeImage.GetOutputPort())
    self.origmapper.SetColorWindow(500)
    self.origmapper.SetColorLevel(100)
    self.origmapper.SetZSlice(100)
    #self.origmapper.SetSliceNumber(100)

    # Set mapper for image data
    self.mapper.SetInputConnection(self.resizeSeg.GetOutputPort())
    self.mapper.SetColorWindow(500)
    self.mapper.SetColorLevel(100)
    self.mapper.SetZSlice(100)

    #stenicl mapper
    self.stencilmapper.SetInputConnection(self.stencil.GetOutputPort())
    # Actor
    self.origactor.SetMapper(self.origmapper)
    self.actor.SetMapper(self.mapper)
    self.stencilactor.SetMapper(self.stencilmapper)


    #self.renderer.AddActor(self.origactor)
    #self.renderer.AddActor(self.actor)
    self.renderer.AddActor(self.stencilactor)
    self.refreshRenderWindow()

    return

  def read_dictionary(self):
    # read predefined dictionary of shapes for
    file = open('../Image_Datasets/shape_dic.pkl', 'rb')
    shape_dic = pickle.load(file)
    file.close()
    self.shape_dic = shape_dic
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
  def addLesion(self):

    #add lesion to current zSlice
    img_copy = self.imageArray.copy()

    zSlice = self.mapper.GetZSlice()
    print(zSlice)

    if np.sum(self.thresholdArray[:,:,zSlice]) == 0:
        self.statusBar().showMessage(f"cannot place lesion in segmented region",4000)
    else:
        print(1)
        idx = np.argwhere(self.thresholdArray[:,:,zSlice]==1)

        random_index = [np.random.randint(0,len(self.shape_dic),1),
                        np.random.randint(0,len(idx),1)]
        random_shape = self.shape_dic[random_index[0][0]]
        top_right_pos = idx[random_index[1][0]]
        x1, x2 = top_right_pos[0], top_right_pos[0] + random_shape.shape[0]
        y1, y2 = top_right_pos[1],top_right_pos[1] + random_shape.shape[1]
        image_cut = img_copy[x1:x2, y1:y2,zSlice]
        image_cut[random_shape] = self.brightness

        self.lesion_dic[zSlice] = [image_cut, random_shape, x1, x2, y1, y2]
        # make vtk image data structure for new image with lesion
    self.lesion.CopyStructure(self.reader.GetOutput())
    self.lesion.GetPointData().SetScalars(numpy_to_vtk(num_array=
                                    img_copy.ravel(order='F'),
                                    deep=True))


  def adjustBrightness(self):

    zSlice = self.mapper.GetZSlice()
    if zSlice in self.lesion_dic:

        lesionParams = self.lesion_dic[zSlice]
        x1,x2,y1,y2 = lesionParams[-4:]
        image_cut = lesionParams[0]
        random_shape = lesionParams[1]
        imageData = self.lesion
        imageDataArray = imageData.GetPointData().GetScalars()
        imageArray = vtk_to_numpy(imageDataArray).reshape(imageData.GetDimensions(), order='F')
        image_cut[random_shape] = self.brightness

        imageArray[x1:x2,y1:y2,zSlice] = image_cut

        self.lesion.GetPointData().SetScalars(numpy_to_vtk(num_array=imageArray.ravel(order='F'),
                                    deep=True))

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
  # Add observers for mouse wheel events to scroll through slices
  def wheelForward(self, obj, event):
  	zSlice = self.mapper.GetZSlice()

  	if (zSlice < self.mapper.GetWholeZMax()):
              self.mapper.SetZSlice(zSlice + 1)
              self.origmapper.SetZSlice(zSlice + 1)

  def wheelBackward(self, obj, event):
  	zSlice = self.mapper.GetZSlice()
  	if (zSlice > self.mapper.GetWholeZMin()):
  	    self.mapper.SetZSlice(zSlice - 1)
  	    self.origmapper.SetZSlice(zSlice + 1)

  def keyPressEvent(self, obj, event):
    key = obj.GetKeySym()
    if key == 'n':
        #self.placer.RemoveAllBoundingPlanes()
        #self.placer.SetProjectionNormalToZAxis()

        self.contourWidget.SetInteractor(self.iren)
        self.contourWidget.SetRepresentation(self.contourRep)
        self.contourWidget.On()
        self.contourWidget.SetContinuousDraw(1)

        #create plane which is in line with vtk slices


        #plane1 = vtk.vtkPlane()
        #plane1.SetOrigin(bounds[0],bounds[2],bounds[4])
        #plane1.SetNormal(0,0,1)
        #self.placer.AddBoundingPlane(plane1)

        #plane2 = vtk.vtkPlane()
        #plane2.SetOrigin(bounds[1],bounds[3],bounds[5])
        #plane2.SetNormal(0,0,-1)
        #self.placer.AddBoundingPlane(plane2)
        #self.placer.SetImageActor(self.resizeImage.GetImageActor())
        self.contourRep.SetPointPlacer(self.placer)
        self.contourWidget.SetEnabled(True)
        obj.Start()


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

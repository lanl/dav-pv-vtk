from vtkmodules.vtkCommonDataModel import vtkDataSet
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa

# new module for ParaView-specific decorators.
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain

from paraview import vtk
import numpy as np # needed for interpolation and pi

import math
import copy


@smproxy.filter(label="VTS to VTS on Sphere")
@smproperty.input(name="Input")
class VTStoVTSonSphere(VTKPythonAlgorithmBase):

    # Earth radius from https://github.com/nsidc/polarstereo-lonlat-convert-py
    #EARTH_RADIUS_KM = 6378.137

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1)

        # Set the default radiusOffset value to 0
        #self.radiusOffset = 0

        # Set the default columnAtEnd value to 0
        self.columnAtEnd = 0

        # Set the default warpByScalar value to 0
        self.warpByScalar = 0

        # Set the default arrayToWarpBy value to an empty string
        self.arrayToWarpBy = ""
        #self.arrayToWarpBy = "Sea Level Change (m)"

        # Set the default secondArrayToWarpBy value to an empty string
        self.secondArrayToWarpBy = ""

        # Set the default warpScaleFactor value to 0
        self.warpScaleFactor = 0

        # Set the default secondWarpScaleFactor value to 0
        self.secondWarpScaleFactor = 0

        # Set the default firstScalarCutoff value to 0
        self.firstScalarCutoff = 0

        # Set the default warpByTwoScalars value to 0
        self.warpByTwoScalars = 0

        # Set the default warpByLog value to 0
        self.warpByLog = 0

        # Set the default sphereRadius value to 6378.137
        self.sphereRadius = 6378.137

        # Create a list of the array names inside of the input data set
        self._availableArrays = ["Sea Level Change (m)", "test"]

    def FillInputPortInformation(self, port, info):
        info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet")
        return 1

    def FillOutputPortInformation(self, port, info):
        info.Set(vtk.vtkDataObject.DATA_TYPE_NAME(), "vtkStructuredGrid")
        return 1

    #@smproperty.dataarrayselection(name="Arrays")
    #def GetDataArraySelection(self):
        #return self._arrayselection
    '''
    #@smproperty.intvector(name="PhiResolution", default_values=0)
    #@smdomain.intrange(min=0, max=1)
    @smproperty.xml("""
        <IntVectorProperty name="IncreaseRadiusOfTheSphere"
            number_of_elements="1"
            default_values="0"
            command="SetRadiusOffset">
            <BooleanDomain name="bool" />
            <Documentation>If on, increases the radius of the sphere by 10 km
            so that the sphere will not occupy the same space as another
            spherical data set.</Documentation>
        </IntVectorProperty>""")
    def SetRadiusOffset(self, x):
        self.radiusOffset = x
        self.Modified()

    def GetRadiusOffset(self):
        return self.radiusOffset
    '''
    @smproperty.xml("""
        <IntVectorProperty name="AddColumnToOneEnd"
            number_of_elements="1"
            default_values="0"
            command="SetColumnAtEnd">
            <BooleanDomain name="bool" />
            <Documentation>If on, copies the values in the first column of the
            input data set and adds that column to the end of the data set.
            </Documentation>
        </IntVectorProperty>""")
    def SetColumnAtEnd(self, x):
        self.columnAtEnd = x
        print("Set Column At End: ", self.columnAtEnd)
        self.Modified()

    def GetColumnAtEnd(self):
        print("Get Column At End: ", self.columnAtEnd)
        return self.columnAtEnd
    
    @smproperty.xml("""
        <IntVectorProperty name="WarpSphereByScalar"
            number_of_elements="1"
            default_values="0"
            command="SetWarpByScalar">
            <BooleanDomain name="bool" />
            <Documentation>If on, will warp the locations of each point of the
            data set based on the values of the scalar array chosen. The areas
            where the scalar array values are positive will cause the data point
            locations to move further from the center while negative scalar
            values at those data points will cause the data point locations to
            shift towards the center of the sphere.</Documentation>
        </IntVectorProperty>

        <PropertyGroup label="Ordered Widgets">
            <Property name="AddColumnToOneEnd" />
            <Property name="SphereRadiusValue" />
            <Property name="WarpSphereByScalar" />
            <Property name="AvailableScalarArrays" />
            <Property name="WarpScaleFactorValue" />
        </PropertyGroup>""")
    def SetWarpByScalar(self, x):
        self.warpByScalar = x
        self.Modified()
    '''
    @smproperty.xml("""
        <IntVectorProperty name="WarpSphereByScalar"
            number_of_elements="1"
            default_values="0"
            command="SetWarpByScalar">
            <BooleanDomain name="bool" />
            <Documentation>If on, will warp the locations of each point of the
            data set based on the values of the scalar array chosen. The areas
            where the scalar array values are positive will cause the data point
            locations to move further from the center while negative scalar
            values at those data points will cause the data point locations to
            shift towards the center of the sphere.</Documentation>
        </IntVectorProperty>""")
    def SetWarpByScalar(self, x):
        self.warpByScalar = x
        self.Modified()
    '''
    def GetWarpByScalar(self):
        return self.warpByScalar

    # Create a slider in the UI for the user to specify the warp scale factor with its
    # range fetched at runtime. For int values,
    # use `intvector` and `IntRangeDomain` instead of the double variants used
    # below.
    @smproperty.doublevector(name="WarpScaleFactorValue", information_only="1")
    def GetValueRange(self):
        print("getting range: (0, 20000)")
        return (0, 20000)

    @smproperty.doublevector(name="WarpScaleFactor", default_values=[0.0])
    @smdomain.xml(\
        """<DoubleRangeDomain name="range" default_mode="mid">
                <RequiredProperties>
                    <Property name="WarpScaleFactorValue" function="RangeInfo" />
                </RequiredProperties>
           </DoubleRangeDomain>
        """)
    def SetValue(self, val):
        print("settings value:", val)
        self.warpScaleFactor = val
        self.Modified()
    
    # Create a slider in the UI for the user to specify the radius of the sphere with
    # its range fetched at runtime. For int values,
    # use `intvector` and `IntRangeDomain` instead of the double variants used
    # below.
    @smproperty.doublevector(name="SphereRadiusValue", information_only="1")
    def GetRadiusValueRange(self):
        print("getting range: (0, 20000)")
        return (0, 20000)

    @smproperty.doublevector(name="SphereRadius", default_values=[6378.137])
    @smdomain.xml(\
        """<DoubleRangeDomain name="range" default_mode="mid">
                <RequiredProperties>
                    <Property name="SphereRadiusValue" function="RangeInfo" />
                </RequiredProperties>
           </DoubleRangeDomain>
        """)
    def SetRadiusValue(self, val):
        print("settings value:", val)
        self.sphereRadius = val
        self.Modified()
    
    @smproperty.stringvector(name="AvailableScalarArrays", information_only="1")
    def GetAvailableArrays(self):
        return (self._availableArrays)
        #return (self.arrayToWarpBy)

    @smproperty.stringvector(name="ScalarArrayToWarpBy", number_of_elements="1")
    @smdomain.xml(\
        """ <StringListDomain name="axisChoice">
                <RequiredProperties>
                    <Property name="AvailableScalarArrays"
                        function="AxisSelection"/>
                </RequiredProperties>
        </StringListDomain>
        """)
    def SetAxis(self, val):
        print("Setting ", val)
        self.arrayToWarpBy = val
        self.Modified()

    @smproperty.stringvector(name="AvailableSecondScalarArrays", information_only="1")
    def GetSecondAvailableArrays(self):
        return (self._availableArrays)
        #return (self.arrayToWarpBy)

    @smproperty.stringvector(name="SecondScalarArrayToWarpBy", number_of_elements="1")
    @smdomain.xml(\
        """ <StringListDomain name="arrayChoice">
                <RequiredProperties>
                    <Property name="AvailableSecondScalarArrays"
                        function="ArraySelection"/>
                </RequiredProperties>
        </StringListDomain>
        """)
    def SetSecondArray(self, val):
        print("Setting ", val)
        self.secondArrayToWarpBy = val
        self.Modified()

    # Create a slider in the UI for the user to specify the second warp scale factor with its
    # range fetched at runtime. For int values,
    # use `intvector` and `IntRangeDomain` instead of the double variants used
    # below.
    @smproperty.doublevector(name="SecondWarpScaleFactorValue", information_only="1")
    def GetSecondValueRange(self):
        print("getting range: (0, 20000)")
        return (0, 20000)

    @smproperty.doublevector(name="SecondWarpScaleFactor", default_values=[0.0])
    @smdomain.xml(\
        """<DoubleRangeDomain name="range" default_mode="mid">
                <RequiredProperties>
                    <Property name="SecondWarpScaleFactorValue" function="RangeInfo" />
                </RequiredProperties>
           </DoubleRangeDomain>
        """)
    def SetSecondValue(self, val):
        print("settings value:", val)
        self.secondWarpScaleFactor = val
        self.Modified()

    # Create a slider in the UI for the user to specify the cutoff value of the first scalar field
    # that will determine when the sphere will be warped by the first scalar field and when it will
    # be warped by the second scalar field. The range
    # is fetched at runtime. For int values,
    # use `intvector` and `IntRangeDomain` instead of the double variants used
    # below.
    @smproperty.doublevector(name="FirstScalarCutoffValue", information_only="1")
    def GetCutoffValueRange(self):
        print("getting range: (-1000, 1000)")
        return (-1000, 1000)

    @smproperty.doublevector(name="FirstScalarArrayCutoff", default_values=[0.0])
    @smdomain.xml(\
        """<DoubleRangeDomain name="range" default_mode="mid">
                <RequiredProperties>
                    <Property name="FirstScalarCutoffValue" function="RangeInfo" />
                </RequiredProperties>
           </DoubleRangeDomain>
        """)
    def SetCutoffValue(self, val):
        print("settings value:", val)
        self.firstScalarCutoff = val
        self.Modified()

    @smproperty.xml("""
        <IntVectorProperty name="WarpByTwoScalars"
            number_of_elements="1"
            default_values="0"
            command="SetWarpByTwoScalars">
            <BooleanDomain name="bool" />
            <Documentation>If on, warps the sphere based on two scalar fields.
            The scalar field that the sphere is warped by is determined based on the
            first scalar field's values at that location on the sphere and whether
            it is above the cutoff value or not. If the first scalar field's value
            is above the cutoff value set by the user, then the sphere will be warped
            by the first scalar field, otherwise it will be warped by the second scalar field.
            </Documentation>
        </IntVectorProperty>""")
    def SetWarpByTwoScalars(self, x):
        self.warpByTwoScalars = x
        print("Set Warp By Two Scalars: ", self.warpByTwoScalars)
        self.Modified()

    def GetWarpByTwoScalars(self):
        print("Get Warp By Two Scalars: ", self.warpByTwoScalars)
        return self.warpByTwoScalars

    # Create a checkbox that, when checked, will make the warp by scalar warp on a 
    # logarithmic scale.
    @smproperty.xml("""
        <IntVectorProperty name="WarpByLogScale"
            number_of_elements="1"
            default_values="0"
            command="SetWarpByLog">
            <BooleanDomain name="bool" />
            <Documentation>If on, warps the data set based on a logarithmic scale
            (using log base 10) rather than a linear scale.
            </Documentation>
        </IntVectorProperty>""")
    def SetWarpByLog(self, x):
        self.warpByLog = x
        print("Set Warp By Log: ", self.warpByLog)
        self.Modified()

    def GetWarpByLog(self):
        print("Get Warp By Log: ", self.warpByLog)
        return self.warpByLog

    # This function will take in an array and find and return one higher
    # longitude value (as a whole degree) than the last element in the array
    def FindNextHighestLonValue(self, arr):
        # Get the highest value in the array arr and truncate the decimal places
        max_whole_lon_val = math.trunc(max(arr))

        # Return the truncated highest longitude value in the array plus one
        return max_whole_lon_val + 1
        # for some reason, returned 181, so I made it return max_whole_lon_val
        # without the plus one for now
        #return max_whole_lon_val


    def AddColumnToEnd(self, inputDataSet):
        # Create the new input data set
        newInputDataSet = vtk.vtkStructuredGrid()

        newPoints = vtk.vtkPoints()
        numPoints = inputDataSet.GetNumberOfPoints()
        print("input data Number of points", numPoints)

        num_arrays = inputDataSet.GetPointData().GetNumberOfArrays()
        print("Number of arrays:", num_arrays)

        # Get the dimensions of the input dataset
        input_dimensions = inputDataSet.GetDimensions()

        print("Dimensions:")
        print(input_dimensions[0])   # should be 1025
        print(input_dimensions[1])   # should be 512
        print(input_dimensions[2])   # should be 1
        x_dimension = input_dimensions[0]
        y_dimension = input_dimensions[1]
        z_dimension = input_dimensions[2]

        original_lon_points = []
        for i in range(0, numPoints):
            original_coord = inputDataSet.GetPoint(i)
            x,y = original_coord[:2]
            original_lon_points.append(x)

        next_lon_value = self.FindNextHighestLonValue(original_lon_points)
        # Should print 360 for the next_lon_value
        print("Next highest longitude value: ", next_lon_value)

        latPoints = []
        lonPoints = []

        for i in range(0, numPoints):
            coord = inputDataSet.GetPoint(i)
            x0, y0 = coord[:2]
            latPoints.append(y0)
            lonPoints.append(x0)

        latPoints = copy.deepcopy(np.array(latPoints))
        lonPoints = copy.deepcopy(np.array(lonPoints))

        # Test out adding a column of y_dimension values of 360 degrees
        # to my longitude array and add a copy of the first
        # column of my latitude array to the end of the array


        # Add the first column of the latitude array to the end
        # of the latitude array
        latPoints=latPoints.reshape(y_dimension,x_dimension)
        ALat = copy.deepcopy(latPoints[0:y_dimension,:])
        BLat = copy.deepcopy(latPoints[0:y_dimension,0:1])
        NewLatArray = np.hstack((ALat, BLat))


        latPoints=latPoints.reshape(y_dimension * x_dimension,)
        NewLatArray=NewLatArray.reshape(y_dimension * (x_dimension + 1),)


        # Add a column of 360 degrees to the end of my longitude
        # array
        #lonPoints=lonPoints.reshape(512,1024)
        lonPoints=lonPoints.reshape(y_dimension,x_dimension)
        #ALon = lonPoints[0:512,:]
        #ALon = copy.deepcopy(lonPoints[0:512,:])
        ALon = copy.deepcopy(lonPoints[0:y_dimension,:])
        #BLon = np.full((512, 1), 360.0)
        #BLon = np.full((y_dimension, 1), 360.0)
        BLon = np.full((y_dimension, 1), next_lon_value)
        NewLonArray = np.hstack((ALon, BLon))


        #lonPoints=lonPoints.reshape(524288,)
        lonPoints=lonPoints.reshape(y_dimension * x_dimension,)
        #NewLonArray=NewLonArray.reshape(524800,)
        NewLonArray=NewLonArray.reshape(y_dimension * (x_dimension + 1),)


        # Check some of the points in the NewLonArray
        #print(NewLonArray[1023])
        #print(NewLonArray[1024])
        print(NewLonArray[x_dimension])
        #print(NewLonArray.shape)

        #for i in range(0,numPoints):
        #for i in range(0,524800):
        for i in range(0,y_dimension * (x_dimension + 1)):
            #mynewcoord = (lonPoints[i], latPoints[i], 0)
            mynewcoord = (NewLonArray[i], NewLatArray[i], 0)
            newPoints.InsertNextPoint(mynewcoord)


        # Loop through each of the scalar arrays in the dataset, add the first
        # column to the end and add the modified array to the new input dataset
        for j in range(0, num_arrays):
            dat = vtk.vtkFloatArray()
            rawdat = []
            ivals = inputDataSet.GetPointData().GetArray(j)
            # populates rawdat list with input points
            for i in range(0, numPoints):
                rawdat.append(ivals.GetValue(i))
            print("rawdat number of points", len(rawdat))
            # Input the values from the altered numpy array into a new vtk
            # float array
            dat.SetName(ivals.GetName())
            newNumPoints = 0
            # Test out the reshape function to see if that is messing up
            # my new input (it seems to work fine after running this code)
            #nprawdat = np.array(rawdat)
            nprawdat = copy.deepcopy(np.array(rawdat))
            nprawdat=nprawdat.reshape(y_dimension,x_dimension)
            nprawdat=nprawdat.reshape(y_dimension * x_dimension,)


            # Test out adding a new column to my rawdat array made up of
            # 512 ones to see if it messes up my code
            nprawdat=nprawdat.reshape(y_dimension,x_dimension)
            A = copy.deepcopy(nprawdat[0:y_dimension,:])
            B = nprawdat[0:y_dimension,0:1]
            newnprawdat = np.hstack((A, B))


            nprawdat=nprawdat.reshape(y_dimension * x_dimension,)
            newnprawdat=newnprawdat.reshape(y_dimension * (x_dimension + 1),)

            # populate newpoints vtkFloatArray
            for i in newnprawdat:
                dat.InsertNextValue(i)


            # Add the array to the new input dataset
            newInputDataSet.GetPointData().AddArray(dat)


        # Check some of the points in the vtk point array
        #print(newPoints.GetPoint(1024))
        print(newPoints.GetPoint(x_dimension))
        #print(newPoints.GetPoint(524799))


        #newInputDataSet.GetPointData().AddArray(dat)
        #newInputDataSet.SetDimensions(1024,512,1)
        #newInputDataSet.SetDimensions(1025,512,1)
        newInputDataSet.SetDimensions((x_dimension + 1),y_dimension,z_dimension)
        newInputDataSet.SetPoints(newPoints)

        return newInputDataSet


    def RequestData(self, request, inInfo, outInfo):

        print("I am running RequestData")

        # Earth radius from https://github.com/nsidc/polarstereo-lonlat-convert-py
        #EARTH_RADIUS_KM = 6378.137

        WARP_SCALE_FACTOR = self.warpScaleFactor
        SECOND_WARP_SCALE_FACTOR = self.secondWarpScaleFactor

        # get the first input.
        inputDataSet0 = dsa.WrapDataObject(vtkDataSet.GetData(inInfo[0]))

        '''
        # Get the list of array names from the input data set and assign that list to the
        # availableArrays variable
        num_arrays = inputDataSet0.GetPointData().GetNumberOfArrays()

        # Create a list of the array names inside of the input data set
        array_list = []
        for i in range(num_arrays):
            array_name = inputDataSet0.GetPointData().GetArray(i).GetName()
            array_list.append(array_name)
        self._availableArrays = array_list
        '''

        # Check that the Add Column To One End checkbox is checked, and if it
        # is, then alter the input data set so that it has a copy of the first
        # column of values added to the end
        if (self.GetColumnAtEnd() == True):
            newDataSet = self.AddColumnToEnd(inputDataSet0)
        else:
            #newDataSet = copy.deepcopy(inputDataSet0)
            newDataSet = inputDataSet0

        # compute a value.
        #data = inputDataSet0.PointData["Sea Level Change (m)"] / 2.0
        #pointX, pointY, pointZ = inputDataSet0.GetPoint(0)

        newPoints = vtk.vtkPoints()
        #numPoints = inputDataSet0.GetNumberOfPoints()
        numPoints = newDataSet.GetNumberOfPoints()
        
        #num_arrays = inputDataSet0.GetPointData().GetNumberOfArrays()
        num_arrays = newDataSet.GetPointData().GetNumberOfArrays()
        print("Number of arrays:", num_arrays)
        
        # Get the dimensions of the input dataset
        #input_dimensions = inputDataSet0.GetDimensions()


        #print("Dimensions:")
        #print(input_dimensions[0])   # should be 1025
        #print(input_dimensions[1])   # should be 512
        #print(input_dimensions[2])   # should be 1

        #x_dimension = input_dimensions[0]
        #y_dimension = input_dimensions[1]
        #z_dimension = input_dimensions[2]

        #print("Increase Radius of the Sphere: ", self.GetRadiusOffset())
        print("Add a column to the end: ", self.GetColumnAtEnd())
        print("Warp Sphere by a Scalar Array: ", self.GetWarpByScalar())

        #radius = EARTH_RADIUS_KM
        radius = self.sphereRadius

        # Check that the Increase Radius Of The Sphere checkbox is checked, and
        # if it is, then add 10 km to the radius variable
        #if (self.GetRadiusOffset() == True):
            #radius = radius + 10

        # Check that the Add Column To One End checkbox is checked, and if it
        # is, then alter the input data set so that it has a copy of the first
        # column of values added to the end
        #if (self.GetColumnAtEnd() == True):
            #inputDataSet0 = self.AddColumnToEnd(inputDataSet0)
            #inputDataSet1 = self.AddColumnToEnd(inputDataSet0)

        # Get the dimensions of the input dataset
        #input_dimensions = inputDataSet0.GetDimensions()
        input_dimensions = newDataSet.GetDimensions()


        print("Dimensions:")
        print(input_dimensions[0])   # should be 1025
        print(input_dimensions[1])   # should be 512
        print(input_dimensions[2])   # should be 1

        x_dimension = input_dimensions[0]
        y_dimension = input_dimensions[1]
        z_dimension = input_dimensions[2]


        # Check that the Add Column To One End checkbox is checked, and if it
        # is, then alter the input data set so that it has a copy of the first
        # column of values added to the end
        #if (self.GetColumnAtEnd() == True):
            #newDataSet = self.AddColumnToEnd(inputDataSet0)
        #else:
            #newDataSet = copy.deepcopy(inputDataSet0)
            #newDataSet = inputDataSet0

        #numPoints = curPts.GetSize()
        #numPoints = newDataSet.GetNumberOfPoints()

        # Check whether the Warp By Two Scalars checkbox is checked, and if it
        # is, then warp the sphere based on two scalar fields with the sphere
        # being warped by the first scalar field when the first scalar field's
        # value is above the cutoff and warp by the second scalar field when the
        # first scalar field's value is below the cutoff.
        if (self.GetWarpByTwoScalars() == False):

            # Check that the Warp Sphere By Scalar checkbox is checked, and if it
            # is, then alter the input data set so that it is warped by the scalar
            # value the user chooses
            if (self.GetWarpByScalar() == True) and (self.GetWarpByLog() == False):
                print("Warp Sphere By Scalar is checked.")
                #newDataSet = self.WarpSphereByScalar(inputDataSet0,WARP_SCALE_FACTOR)
                # Get the scalar values of the first scalar array of the input dataset
                # to warp the sphere by (change this later so that the user can choose
                # which scalar to warp by)
                #oldPointVals = inputDataSet0.GetPointData().GetArray(0)    # sea level change data
                #oldPointVals = inputDataSet0.GetPointData().GetArray("Sea Level Change (m)")    # sea level change data
                oldPointVals = newDataSet.GetPointData().GetArray(self.arrayToWarpBy)

                # Warp the points in newPoints based on the warp scale factor and the
                # values at each of those points in the scalar array chosen
                for i in range(0, numPoints):
                    coord = newDataSet.GetPoint(i)
                    rX, rY, rZ = coord[:3]
                    lat = rY * np.pi / 180
                    lon = rX * np.pi / 180
                    x = radius * np.cos(lat) * np.cos(lon)
                    y = radius * np.cos(lat) * np.sin(lon)
                    z = radius * np.sin(lat)
                    oldRadius = np.sqrt(np.dot([x, y, z], [x, y, z]))
                    #newRadius = oldRadius + (oldPointVals.GetValue(i) * warp_scale_factor)
                    newRadius = oldRadius + (oldPointVals.GetValue(i) * WARP_SCALE_FACTOR)
                    radiusRatio = newRadius/oldRadius
                    #print("Hello world!")

                    #rX = rX * (radius_ratio)
                    warpedX = x * (radiusRatio)
                    #rY = rY * (radius_ratio)
                    warpedY = y * (radiusRatio)
                    #rZ = rZ * (radius_ratio)
                    warpedZ = z * (radiusRatio)
                    newPoints.InsertPoint(i,warpedX,warpedY,warpedZ)

            # The warp by scalar and the warp by log checkboxes are checked
            elif (self.GetWarpByScalar() == True) and (self.GetWarpByLog() == True):
                print("Warp Sphere By Scalar is checked.")
                print("Warp By Log is checked.")
                # Get the scalar values of the scalar array of the input dataset the user
                # chooses to warp the sphere by
                oldPointVals = newDataSet.GetPointData().GetArray(self.arrayToWarpBy)

                # Warp the points in newPoints based on the warp scale factor and the
                # log base 10 values at each of those points in the scalar array chosen
                for i in range(0, numPoints):
                    coord = newDataSet.GetPoint(i)
                    rX, rY, rZ = coord[:3]
                    lat = rY * np.pi / 180
                    lon = rX * np.pi / 180
                    x = radius * np.cos(lat) * np.cos(lon)
                    y = radius * np.cos(lat) * np.sin(lon)
                    z = radius * np.sin(lat)
                    oldRadius = np.sqrt(np.dot([x, y, z], [x, y, z]))
                    # Scalar values are positive
                    if (oldPointVals.GetValue(i) >= 0.00000001):
                        newRadius = oldRadius + (math.log10(oldPointVals.GetValue(i) + 1) * WARP_SCALE_FACTOR)
                    # Scalar values are negative
                    elif (oldPointVals.GetValue(i) <= -0.00000001):
                        newRadius = oldRadius - (math.log10(abs(oldPointVals.GetValue(i)) + 1) * WARP_SCALE_FACTOR)
                        #print(newRadius, end = ",")
                    # Scalar values are around zero
                    else:
                        #newRadius = oldRadius + (oldPointVals.GetValue(i) * WARP_SCALE_FACTOR)
                        newRadius = oldRadius
                    radiusRatio = newRadius/oldRadius

                    warpedX = x * (radiusRatio)
                    warpedY = y * (radiusRatio)
                    warpedZ = z * (radiusRatio)
                    newPoints.InsertPoint(i,warpedX,warpedY,warpedZ)
            else:
                # Go through each of the points and map them to their
                # sphere coordinates in Cartesian space
                for i in range(0, numPoints):
                    coord = newDataSet.GetPoint(i)
                    #coord = curPts.GetPoint(i)
                    x0, y0, z0 = coord[:3]
                    lat = y0 * np.pi / 180
                    lon = x0 * np.pi / 180
                    x = radius * np.cos(lat) * np.cos(lon)
                    y = radius * np.cos(lat) * np.sin(lon)
                    z = radius * np.sin(lat)
                    if ((x == 0) and (y == 0) and (z == 0)):
                        print(x0, y0, z0)
                    newPoints.InsertPoint(i,x,y,z)

        # The Warp By Two Scalars is checked
        else:
            print("Warp Sphere By Two Scalars is checked.")
            # Get the first scalar field values of the scalar array of the input dataset 
            # the user chooses to warp the sphere by
            oldPointVals = newDataSet.GetPointData().GetArray(self.arrayToWarpBy)

            # Get the second scalar field values of the scalar array of the input dataset 
            # the user chooses to warp the sphere by
            oldPointVals2 = newDataSet.GetPointData().GetArray(self.secondArrayToWarpBy)

            larger_radius = radius + 100

            # Warp the points in newPoints based on the warp scale factors and the
            # values at each of those points in the scalar arrays chosen
            for i in range(0, numPoints):
                coord = newDataSet.GetPoint(i)
                rX, rY, rZ = coord[:3]
                lat = rY * np.pi / 180
                lon = rX * np.pi / 180
                if (oldPointVals.GetValue(i) >= self.firstScalarCutoff):
                    x = larger_radius * np.cos(lat) * np.cos(lon)
                    y = larger_radius * np.cos(lat) * np.sin(lon)
                    z = larger_radius * np.sin(lat)
                    oldRadius = np.sqrt(np.dot([x, y, z], [x, y, z]))
                    #newRadius = oldRadius + (oldPointVals.GetValue(i) * warp_scale_factor)
                    newRadius = oldRadius + (oldPointVals.GetValue(i) * WARP_SCALE_FACTOR)

                else:
                    x = radius * np.cos(lat) * np.cos(lon)
                    y = radius * np.cos(lat) * np.sin(lon)
                    z = radius * np.sin(lat)
                    oldRadius = np.sqrt(np.dot([x, y, z], [x, y, z]))
                    #newRadius = oldRadius + (oldPointVals.GetValue(i) * warp_scale_factor)
                    newRadius = oldRadius + (oldPointVals2.GetValue(i) * SECOND_WARP_SCALE_FACTOR)

                radiusRatio = newRadius/oldRadius
                #print("Hello world!")

                #rX = rX * (radius_ratio)
                warpedX = x * (radiusRatio)
                #rY = rY * (radius_ratio)
                warpedY = y * (radiusRatio)
                #rZ = rZ * (radius_ratio)
                warpedZ = z * (radiusRatio)
                newPoints.InsertPoint(i,warpedX,warpedY,warpedZ)

        # Create the output data set
        outputDataSet = vtk.vtkStructuredGrid.GetData(outInfo)


        # Loop through each of the scalar arrays in the dataset
        # and add the array to the output dataset
        for j in range(0, num_arrays):
            #ivals = inputDataSet0.GetPointData().GetArray(j)
            ivals = newDataSet.GetPointData().GetArray(j)
            ca = vtk.vtkFloatArray()
            ca.SetName(ivals.GetName())
            ca.SetNumberOfComponents(1)
            ca.SetNumberOfTuples(numPoints)

            # Print out the number of points
            print("Number of points: ", newPoints.GetNumberOfPoints())
        

            print("Output data set: ", outputDataSet)


            #add the new array to the output
            outputDataSet.GetPointData().AddArray(ca)


            #copy the values over element by element
            for i in range(0, numPoints):
                ca.SetValue(i, ivals.GetValue(i))

            # Try printing out the value at the 1025th point
            print("Value at point 1024: ", ivals.GetValue(1024))
            print("Value at point 1024 in ca: ", ca.GetValue(1024))

        outputDataSet.SetDimensions(x_dimension,y_dimension,z_dimension)
        outputDataSet.SetPoints(newPoints)

        print(outputDataSet)


        return 1

    def RequestInformation(self, request, inInfo, outInfo):
        print("I am running RequestInformation")
        #print("I am running RequestInformation")

        # get the first input.
        inputDataSet0 = dsa.WrapDataObject(vtkDataSet.GetData(inInfo[0]))

        # Get the list of array names from the input data set and assign that list to the
        # availableArrays variable
        num_arrays = inputDataSet0.GetPointData().GetNumberOfArrays()

        # Create a list of the array names inside of the input data set
        array_list = []
        for i in range(num_arrays):
            array_name = inputDataSet0.GetPointData().GetArray(i).GetName()
            array_list.append(array_name)
        self._availableArrays = array_list
        
        # Get the dimensions of the input dataset
        inputData = dsa.WrapDataObject(vtkDataSet.GetData(inInfo[0]))
        input_dimensions = inputData.GetDimensions()
        x_dimension = input_dimensions[0]
        y_dimension = input_dimensions[1]
        z_dimension = input_dimensions[2]


        executive = self.GetExecutive()
        outInfo = executive.GetOutputInformation(0)
        outInfo.Set(executive.WHOLE_EXTENT(), 0, x_dimension, 0, (y_dimension - 1), 0, 0)
        return 1

    def RequestUpdateExtent(self, request, inInfo, outInfo):
        print("I am running RequestUpdateExtent")

        
        # Get the dimensions of the input dataset
        inputData = dsa.WrapDataObject(vtkDataSet.GetData(inInfo[0]))
        input_dimensions = inputData.GetDimensions()
        x_dimension = input_dimensions[0]
        y_dimension = input_dimensions[1]
        z_dimension = input_dimensions[2]


        executive = self.GetExecutive()
        inInfo = executive.GetInputInformation(0, 0)
        inInfo.Set(executive.UPDATE_EXTENT(), 0, (x_dimension - 1), 0, (y_dimension - 1), 0, 0)
        return 1


from vtkmodules.vtkCommonDataModel import vtkDataSet
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkGeovisCore import vtkGeoProjection, vtkGeoTransform
try:
    from pyproj import Transformer
except:
    print("The pyproj library is not installed. Install the pyproj library to be able to put the data set on a sphere.")
#from pyproj import Transformer

#import pyproj
import math
import copy

from vtk.util import numpy_support

# new module for ParaView-specific decorators.
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain

from paraview import vtk
import numpy as np #needed for interpolation and pi

@smproxy.filter(label="Latitude Longitude Reproject")
@smproperty.input(name="Input")
class GeoLatLonReproject(VTKPythonAlgorithmBase):

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1)

        # Set the default realMeridian value to 0
        self.realMeridian = 0

        # Set the default columnAtEnd value to 0
        self.columnAtEnd = 0

        # Set the default projection value to an empty string
        self.projection = ""

        # Set the default availableArrays value to an empty string
        self._availableArrays = ""
        
        # Test to see if pyproj can be imported. If it can, include the
        # sphere as a possible projection, otherwise remove the sphere
        # from the possible projections to choose from.
        try:
            from pyproj import Transformer
            #import pyproj
            #print("Pyproj was successfully able to be imported.")
            # Create a list of common map projections to choose from
            self._mapProjectionList = ["Robinson", "Mercator", "Northern Hemisphere Stereographic", "Southern Hemisphere Stereographic", "Lambert Conformal Conic", "Sphere"]
        except:
            print("Pyproj was not able to be imported.")
            # Create a list of common map projections to choose from
            self._mapProjectionList = ["Robinson", "Mercator", "Northern Hemisphere Stereographic", "Southern Hemisphere Stereographic", "Lambert Conformal Conic"]

        # Create a list of common map projections to choose from
        #self._mapProjectionList = ["Robinson", "Mercator", "Northern Hemisphere Stereographic", "Southern Hemisphere Stereographic", "Lambert Conformal Conic", "Sphere"]
        
        # Set the default sphereRadius value to 0
        #self.sphereRadius = 0

    def FillInputPortInformation(self, port, info):
        info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet")
        return 1

    def FillOutputPortInformation(self, port, info):
        info.Set(vtk.vtkDataObject.DATA_TYPE_NAME(), "vtkStructuredGrid")
        return 1

    #@smproperty.doublevector(name="SphereRadiusValue", information_only="1")
    #def

    @smproperty.stringvector(name="AvailableMapProjections", information_only="1")
    def GetAvailableProjections(self):
        return(self._mapProjectionList)

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
        #print("Set Column At End: ", self.columnAtEnd)
        self.Modified()

    def GetColumnAtEnd(self):
        #print("Get Column At End: ", self.columnAtEnd)
        return self.columnAtEnd

    @smproperty.stringvector(name="MapProjections", number_of_elements="1")
    @smdomain.xml(\
        """ <StringListDomain name="projChoice">
                <RequiredProperties>
                    <Property name="AvailableMapProjections"
                        function="projSelection"/>
                </RequiredProperties>
        </StringListDomain>
        """)
    def SetProj(self, val):
        #print("Setting ", val)
        self.projection = val
        self.Modified()

    @smproperty.xml("""
        <IntVectorProperty name="CentralMeridianAtZero"
            number_of_elements="1"
            default_values="0"
            command="SetCentralMeridian">
            <BooleanDomain name="bool" />
            <Documentation>If on, sets the central meridian of the map
            projection to zero degrees longitude.</Documentation>
        </IntVectorProperty>""")
    def SetCentralMeridian(self, x):
        self.realMeridian = x
        self.Modified()

    def GetCentralMeridian(self):
        return self.realMeridian

    # This function will take in an array and the number of elements inside of it
    # and find and return the index of the element in the array that is equal to
    # or approximately equal to the longitude value given
    def FindIndxAtLon(self, arr, arr_dimension,lonVal):
        for i in range(arr_dimension):
            # Check that the element in the array is less than 0.35 away from the 
            # longitude value
            if (abs((lonVal - arr[i])) < 0.35):
                return i
        
        return -1

    def SplitAtLon(self, inputDataSet, lonVal):
        # Create the new input data set
        newInputDataSet = vtk.vtkStructuredGrid()

        newPoints = vtk.vtkPoints()
        numPoints = inputDataSet.GetNumberOfPoints()

        num_arrays = inputDataSet.GetPointData().GetNumberOfArrays()
        #print("Number of arrays:", num_arrays)

        # Get the dimensions of the input dataset
        input_dimensions = inputDataSet.GetDimensions()
        # Create the new input data set
        newInputDataSet = vtk.vtkStructuredGrid()

        newPoints = vtk.vtkPoints()
        numPoints = inputDataSet.GetNumberOfPoints()

        num_arrays = inputDataSet.GetPointData().GetNumberOfArrays()
        #print("Number of arrays:", num_arrays)

        # Get the dimensions of the input dataset
        input_dimensions = inputDataSet.GetDimensions()

        #print("Dimensions:")
        #print(input_dimensions[0])   # should be 1025
        #print(input_dimensions[1])   # should be 512
        #print(input_dimensions[2])   # should be 1
        x_dimension = input_dimensions[0]
        y_dimension = input_dimensions[1]
        z_dimension = input_dimensions[2]

        original_lon_points = []
        for i in range(0, numPoints):
            original_coord = inputDataSet.GetPoint(i)
            x,y = original_coord[:2]
            original_lon_points.append(x)

        lon_indx = self.FindIndxAtLon(original_lon_points, x_dimension, lonVal)
        #print("Index of the x-dimension with a longitude of ", lonVal, ": ", lon_indx, sep ="")

        latPoints = []
        lonPoints = []

        for i in range(0, numPoints):
            coord = inputDataSet.GetPoint(i)
            x0, y0 = coord[:2]
            if x0 > 180:
                x0 = x0 - 360
            latPoints.append(y0)
            lonPoints.append(x0)

        lonPoints = np.array(lonPoints)
        latPoints = np.array(latPoints)
        latPoints=latPoints.reshape(y_dimension, x_dimension)
        lonPoints=lonPoints.reshape(y_dimension, x_dimension)
        
        A = lonPoints[0:y_dimension,0:int(lon_indx + 1)]    # [Rows, Columns]
        B = lonPoints[0:y_dimension,int(lon_indx + 1):]    # Right hand side of the dataset
        NewLonArray = np.hstack((B,A))
        NewLonArray = NewLonArray.reshape(y_dimension * x_dimension,)

        ALat = latPoints[0:y_dimension,0:int(lon_indx)]
        BLat = latPoints[0:y_dimension,int(lon_indx):]
        NewLatArray = np.hstack((BLat, ALat))
        NewLatArray = NewLatArray.reshape(y_dimension * x_dimension,)

        # Take the x, y and z coordinates from the input data's vtk
        # point array and convert them into a Robinson projection
        # before putting the new coordinate values into a list
        RobinsonPoints = []
        for x0,y0 in zip(NewLonArray, NewLatArray):
            RobinsonPoints.append((x0,y0,0))


        # Put the Robinson projection coordinates into the empty vtk
        # point array newPoints
        count = 0
        for i in RobinsonPoints:
            newPoints.InsertNextPoint(i)
            count += 1

        # Loop through each of the scalar arrays in the dataset
        # and add the array to the output dataset
        for j in range(0, num_arrays):
            # Convert the scalar values into a list
            rawdat = []
            ivals = inputDataSet.GetPointData().GetArray(j)
            for i in range(0, numPoints):
                rawdat.append(ivals.GetValue(i))

            # Create a numpy array from the rawdat list and move the right
            # half of the array (the scalar values found at longitudes
            # greater than 180 degrees) over to the left
            nprawdat = np.array(rawdat)
            nprawdat=nprawdat.reshape(y_dimension, x_dimension)

            A = nprawdat[0:y_dimension,0:int(lon_indx)]
            B = nprawdat[0:y_dimension,int(lon_indx):]
            NewArray = np.hstack((B,A))
            NewArray = NewArray.reshape(y_dimension * x_dimension,)

            # Input the values from the altered numpy array into a new vtk
            # float array
            dat = vtk.vtkFloatArray()
            dat.SetName(ivals.GetName())
            for i in NewArray:
                dat.InsertNextValue(i)

            ca = vtk.vtkFloatArray()
            ca.SetName(ivals.GetName())
            ca.SetNumberOfComponents(1)
            ca.SetNumberOfTuples(numPoints)

            #add the new array to the newInputDataSet
            newInputDataSet.GetPointData().AddArray(ca)

            #copy the values over element by element
            for i in range(0, numPoints):
                ca.SetValue(i, dat.GetValue(i))

        newInputDataSet.SetDimensions(x_dimension,y_dimension,z_dimension)
        newInputDataSet.SetPoints(newPoints)
        
        return newInputDataSet
    
    # This function will take in an array and find and return one higher
    # longitude value (as a whole degree) than the last element in the array
    def FindNextHighestLonValue(self, arr):
        # Get the highest value in the array arr and truncate the decimal places
        max_whole_lon_val = math.trunc(max(arr))

        # Return the truncated highest longitude value in the array plus one
        return max_whole_lon_val + 1

    def AddColumnToEnd(self, inputDataSet):
        # Create the new input data set
        newInputDataSet = vtk.vtkStructuredGrid()

        newPoints = vtk.vtkPoints()
        numPoints = inputDataSet.GetNumberOfPoints()
        #print("input data Number of points", numPoints)

        num_arrays = inputDataSet.GetPointData().GetNumberOfArrays()
        #print("Number of arrays:", num_arrays)

        # Get the dimensions of the input dataset
        input_dimensions = inputDataSet.GetDimensions()

        #print("Dimensions:")
        #print(input_dimensions[0])   # should be 1025
        #print(input_dimensions[1])   # should be 512
        #print(input_dimensions[2])   # should be 1
        x_dimension = input_dimensions[0]
        y_dimension = input_dimensions[1]
        z_dimension = input_dimensions[2]

        original_lon_points = []
        for i in range(0, numPoints):
            original_coord = inputDataSet.GetPoint(i)
            x,y = original_coord[:2]
            original_lon_points.append(x)

        #next_lon_value = self.FindNextHighestLonValue(original_lon_points)
        next_lon_value = original_lon_points[0]
        #print("Next highest longitude value: ", next_lon_value)

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
        lonPoints=lonPoints.reshape(y_dimension,x_dimension)
        ALon = copy.deepcopy(lonPoints[0:y_dimension,:])
        BLon = np.full((y_dimension, 1), next_lon_value)
        NewLonArray = np.hstack((ALon, BLon))

        lonPoints=lonPoints.reshape(y_dimension * x_dimension,)
        NewLonArray=NewLonArray.reshape(y_dimension * (x_dimension + 1),)


        # Check some of the points in the NewLonArray
        #print(NewLonArray[x_dimension])
        #print(NewLonArray.shape)

        for i in range(0,y_dimension * (x_dimension + 1)):
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
            #print("rawdat number of points", len(rawdat))
            # Input the values from the altered numpy array into a new vtk
            # float array
            dat.SetName(ivals.GetName())
            newNumPoints = 0
            # Test out the reshape function to see if that is messing up
            # my new input (it seems to work fine after running this code)
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
        #print(newPoints.GetPoint(x_dimension))


        newInputDataSet.SetDimensions((x_dimension + 1),y_dimension,z_dimension)
        newInputDataSet.SetPoints(newPoints)

        return newInputDataSet
    
    def GetProjString(self):
        projString = ""
        if (self.projection == "Robinson"):
            projString = "+proj=robin +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        elif (self.projection == "Mercator"):
            projString = "+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        elif (self.projection == "Northern Hemisphere Stereographic"):
            projString = "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        elif (self.projection == "Southern Hemisphere Stereographic"):
            projString = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        elif (self.projection == "Lambert Conformal Conic"):
            projString = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        elif (self.projection == "Sphere"):
            projString = "+proj=cart +ellps=WGS84"
        return projString

    # Will take in a string representing which hemisphere the user wants to
    # project along with the input data set and will return the part of the
    # input data set that belongs in the hemisphere given.
    def GetHemisphere(self, hemisphere, inputData):
        hemisphereData = vtk.vtkStructuredGrid()
        num_arrays = inputData.GetPointData().GetNumberOfArrays()

        # Get the dimensions of the input data set
        input_dimension = inputData.GetDimensions()
        x_dimension = input_dimension[0]
        y_dimension = input_dimension[1]
        z_dimension = input_dimension[2]

        hemisphereDataPts = vtk.vtkPoints()
        indx = 0
        for i in range(0, inputData.GetNumberOfPoints()):
            coord = inputData.GetPoint(i)
            x0, y0, z0 = coord[:3]
            if (hemisphere == "northern"):
                if y0 >=0:
                    hemisphereDataPts.InsertPoint(indx,x0,y0,0)
                    indx = indx + 1
            else:
                if y0 <= 0:
                    hemisphereDataPts.InsertPoint(indx,x0,y0,0)
                    indx = indx + 1

        new_y_dimension = hemisphereDataPts.GetNumberOfPoints() // x_dimension

        # Loop through each of the scalar arrays in the inputData and add the
        # scalar values in the hemisphere given by the user to the hemisphereData
        # vtk structured grid.
        for j in range(0, num_arrays):
            ivals = inputData.GetPointData().GetArray(j)
            ca = vtk.vtkFloatArray()
            ca.SetName(ivals.GetName())
            ca.SetNumberOfComponents(1)
            ca.SetNumberOfTuples(hemisphereDataPts.GetNumberOfPoints())
            # add the new array to the hemisphereData
            hemisphereData.GetPointData().AddArray(ca)
            # copy the values over element by element
            indx2 = 0
            for k in range(0, inputData.GetNumberOfPoints()):
                coord = inputData.GetPoint(k)
                x, y, z = coord[:3]
                if (hemisphere == "northern"):
                    if y >= 0:
                        ca.SetValue(indx2, ivals.GetValue(k))
                        indx2 = indx2 + 1
                else:
                    if y <= 0:
                        ca.SetValue(indx2, ivals.GetValue(k))
                        indx2 = indx2 + 1

        hemisphereData.SetDimensions(x_dimension, new_y_dimension, z_dimension)
        hemisphereData.SetPoints(hemisphereDataPts)
        return hemisphereData

    # This function will take in a vtk data set and return a vtk points array of the
    # points found inside the data set
    def GetPoints(self, vtkDataSet):
        vtkPts = vtk.vtkPoints()
        try:
            vtkPts = vtkDataSet.GetPoints()
        except:
            #print("The data set is not a vtk structured grid")
            numPoints = vtkDataSet.GetNumberOfPoints()
            for i in range(numPoints):
                point = vtkDataSet.GetPoint(i)
                x, y, z = point[:3]
                vtkPts.InsertPoint(i,x,y,z)
        
        return vtkPts

    def RequestData(self, request, inInfo, outInfo):
        # get the first input.
        inputDataSet1 = dsa.WrapDataObject(vtkDataSet.GetData(inInfo[0]))

        newPoints = vtk.vtkPoints()
        
        num_arrays = inputDataSet1.GetPointData().GetNumberOfArrays()
        #print("Number of arrays:", num_arrays)
        
        #print("Real Meridian: ", self.GetCentralMeridian())

        # If the user chooses the Robinson, Mercator or Lambert Conformal Conic projection,
        # then set the columnAtEnd variable to zero
        #if (((self.projection == "Robinson") or (self.projection == "Mercator") or (self.projection == "Lambert Conformal Conic"))):
             #self.columnAtEnd = 0

        # Check that the Central Meridian At Zero checkbox is checked, and if it
        # is, then alter the input data set so that it is centered at zero
        # degrees longitude. Also check whether the projection is a lambert 
        # conformal conic or a stereographic projection, and if it is one of
        # those projections, then alter the input data set to only include one
        # hemisphere.
        if ((self.GetCentralMeridian() == True) and (self.projection == "Northern Hemisphere Stereographic")):
            wholeInputDataSet = self.SplitAtLon(inputDataSet1,180)
            inputDataSet2 = self.GetHemisphere("northern", wholeInputDataSet) 
        elif (self.projection == "Lambert Conformal Conic"):
            wholeInputDataSet = self.SplitAtLon(inputDataSet1,83)
            inputDataSet2 = self.GetHemisphere("northern", wholeInputDataSet) 
        elif ((self.GetCentralMeridian() == True) and (self.projection == "Southern Hemisphere Stereographic")):
            wholeInputDataSet = self.SplitAtLon(inputDataSet1,180)
            inputDataSet2 = self.GetHemisphere("southern", wholeInputDataSet) 
        elif (self.GetCentralMeridian() == True):
            inputDataSet2 = self.SplitAtLon(inputDataSet1,180)
        elif (self.projection == "Northern Hemisphere Stereographic"):
            inputDataSet2 = self.GetHemisphere("northern", inputDataSet1) 
        elif (self.projection == "Southern Hemisphere Stereographic"):
            inputDataSet2 = self.GetHemisphere("southern", inputDataSet1) 
        else:
            inputDataSet2 = inputDataSet1
        
        if (self.GetColumnAtEnd() == True) and (((self.projection != "Robinson") and (self.projection != "Mercator") and (self.projection != "Lambert Conformal Conic"))):
            inputDataSet0 = self.AddColumnToEnd(inputDataSet2)
        else:
            inputDataSet0 = inputDataSet2
        
        # Get the dimensions of the input dataset
        input_dimensions = inputDataSet0.GetDimensions()

        #print("Dimensions:")
        #print(input_dimensions[0])   # should be 1025
        #print(input_dimensions[1])   # should be 512
        #print(input_dimensions[2])   # should be 1

        x_dimension = input_dimensions[0]
        y_dimension = input_dimensions[1]
        z_dimension = input_dimensions[2]

        numPoints = inputDataSet0.GetNumberOfPoints()

        # Get the longitude value at the first and last columns and calculate
        # central meridian to use for the Robinson projection
        first_col = inputDataSet0.GetPoint(0)
        last_col = inputDataSet0.GetPoint(x_dimension - 1)

        first_lon = first_col[0]
        last_lon = last_col[0]

        mid_lon = first_lon + ((last_lon - first_lon)/2)

        # Convert the central meridian from degrees to radians
        mid_lon_rad = mid_lon * np.pi/180

        # Get the source and destination projections
        geo = vtkGeoTransform()
        ps = vtkGeoProjection()
        pd = vtkGeoProjection()
        pd.SetPROJ4String(self.GetProjString())

        if (self.projection == "sphere"):
            ps.SetPROJ4String("+proj=lonlat +ellps=WGS84")

        geo.SetSourceProjection(ps)
        geo.SetDestinationProjection(pd)
        geopts = vtk.vtkPoints()

        # populate oldPoints with the original points of the inputDataSet
        # and then transform those points into a Robinson projection and
        # put them in newPoints
        #oldPointsArray = inputDataSet0.GetPoints()
        oldPointsArray = self.GetPoints(inputDataSet0)
        #if (((self.projection == "Robinson") or (self.projection == "Mercator") or (self.projection == "Sphere")) and (self.GetCentralMeridian() != True) and (self.GetColumnAtEnd() != True)):
        if ((((self.projection == "Robinson") or (self.projection == "Mercator")) and (self.GetCentralMeridian() != True)) or ((self.projection == "Sphere") and (self.GetColumnAtEnd() != True) and (self.GetCentralMeridian() != True))):
            #oldPoints = inputDataSet0.VTKObject.GetPoints()
            oldPoints = self.GetPoints(inputDataSet0.VTKObject)
        else:
            #oldPoints = inputDataSet0.GetPoints()
            oldPoints = self.GetPoints(inputDataSet0)
        #print(oldPoints)

        # If the projection is set to Sphere, then loop through each point one
        # at a time to convert the coordinates.
        '''
        if (self.projection == "Sphere"):
            ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
            #lla = pyproj.Proj(proj='lonlat', ellps='WGS84', datum='WGS84')
            lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

            # populate newPoints with Sphere points with the assumption that
            # the data is at the same elevation as the sphere (the sphere is
            # smooth without any warping)
            for i in range(0, numPoints):
                coord = inputDataSet0.GetPoint(i)
                #x0, y0, z0 = coord[:3]
                x, y, z = coord[:3]
                #x,y,z = pyproj.transform(lla,ecef,x0,y0,0,radians=False)
                x1,y1,z1 = pyproj.transform(lla,ecef,x,y,0,radians=False)
                #x1,y1,z1 = pyproj.transform(lla,ecef,180,0,0,radians=False)
                newPoints.InsertPoint(i,x,y,z)

        else:
            geo.TransformPoints(oldPoints, newPoints)
        '''
        
        if (self.projection == "Sphere"):
            transproj = Transformer.from_crs({'proj':'latlong', 'ellps':'WGS84', 'datum':'WGS84'}, {'proj':'geocent', 'ellps':'WGS84', 'datum':'WGS84'}, always_xy=True)
            npArrayOldPoints = numpy_support.vtk_to_numpy(oldPoints.GetData())
            listOldPoints = list(npArrayOldPoints)

            listNewPoints = list(transproj.itransform(listOldPoints))
            npArrayNewPoints = np.array(listNewPoints)
            vtkArrayNewPoints = numpy_support.numpy_to_vtk(npArrayNewPoints, deep=True, array_type=vtk.VTK_FLOAT)

            for i in range(vtkArrayNewPoints.GetNumberOfTuples()):
                newPoints.InsertNextPoint(vtkArrayNewPoints.GetTuple3(i))
        else:
            geo.TransformPoints(oldPoints, newPoints)

        # Create the output data set
        outputDataSet = vtk.vtkStructuredGrid.GetData(outInfo)


        # Loop through each of the scalar arrays in the dataset
        # and add the array to the output dataset
        for j in range(0, num_arrays):
            ivals = inputDataSet0.GetPointData().GetArray(j)
            ca = vtk.vtkFloatArray()
            ca.SetName(ivals.GetName())
            ca.SetNumberOfComponents(1)
            ca.SetNumberOfTuples(numPoints)

            # Print out the number of points
            #print("Number of points: ", newPoints.GetNumberOfPoints())

            #print("Output data set: ", outputDataSet)

            #add the new array to the output
            outputDataSet.GetPointData().AddArray(ca)

            #copy the values over element by element
            for i in range(0, numPoints):
                ca.SetValue(i, ivals.GetValue(i))

            # Try printing out the value at the 1025th point
            #print("Value at point 1024: ", ivals.GetValue(1024))
            #print("Value at point 1024 in ca: ", ca.GetValue(1024))

        outputDataSet.SetDimensions(x_dimension,y_dimension,z_dimension)
        outputDataSet.SetPoints(newPoints)

        #print(outputDataSet)
        

        return 1

    def RequestInformation(self, request, inInfo, outInfo):
        #print("I am running RequestInformation")

        # get the first input
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

        # Get the dimensions of the input data set
        inputData = dsa.WrapDataObject(vtkDataSet.GetData(inInfo[0]))
        input_dimensions = inputData.GetDimensions()
        x_dimension = input_dimensions[0]
        y_dimension = input_dimensions[1]
        z_dimension = input_dimensions[2]

        executive = self.GetExecutive()
        outInfo = executive.GetOutputInformation(0)
        outInfo.Set(executive.WHOLE_EXTENT(), 0, x_dimension, 0, (y_dimension - 1), 0, (z_dimension - 1))
        return 1

    def RequestUpdateExtent(self, request, inInfo, outInfo):
        #print("I am running RequestUpdateExtent")

        # Get the dimensions of the input data set
        inputData = dsa.WrapDataObject(vtkDataSet.GetData(inInfo[0]))
        input_dimensions = inputData.GetDimensions()
        x_dimension = input_dimensions[0]
        y_dimension = input_dimensions[1]
        z_dimension = input_dimensions[2]

        executive = self.GetExecutive()
        inInfo = executive.GetInputInformation(0, 0)
        inInfo.Set(executive.UPDATE_EXTENT(), 0, (x_dimension - 1), 0, (y_dimension - 1), 0, (z_dimension - 1))
        return 1 

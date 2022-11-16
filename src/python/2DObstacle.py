#> \file
#> \author Chris Bradley
#> \brief This is an example program which solves a the 2D Navier-Stokes flow around an obstacle  using OpenCMISS calls.
#>

#================================================================================================================================
#  Start Program
#================================================================================================================================


LINEAR = 1
QUADRATIC = 2
CUBIC = 3
HERMITE = 4

NOTHING = 1
VELOCITY = 2
PRESSURE = 3
REFPRESSURE = 4

numberOfDimensions = 2

numberOfSolidXElements = 2
numberOfSolidYElements = 8
numberOfSolidZElements = 0
numberOfFluidX1Elements = 8
numberOfFluidX2Elements = 20
numberOfFluidYElements = 6
numberOfFluidZ1Elements = 0
numberOfFluidZ2Elements = 0

elementSize = 0.1
solidXSize = numberOfSolidXElements*elementSize
solidYSize = numberOfSolidYElements*elementSize
solidZSize = numberOfSolidZElements*elementSize
fluidX1Size = numberOfFluidX1Elements*elementSize
fluidX2Size = numberOfFluidX2Elements*elementSize
fluidYSize = numberOfFluidYElements*elementSize
fluidZ1Size = numberOfFluidZ1Elements*elementSize
fluidZ2Size = numberOfFluidZ2Elements*elementSize

velocityInterpolation = QUADRATIC
pressureInterpolation = LINEAR

#RBS = False
RBS = True

outputFrequency = 1 # Result output frequency

setupOutput = True
progressDiagnostics = True
debugLevel = 3

startTime = 0.0
stopTime  = 14.0
timeStep  = 0.1

# Inlet velocity parameters
A = 0.5
B = 2.0
C = -0.5

# Material properties
# NOTE: USE OF SI UNITS unless comment
# Low density fluid, rubber-like solid
fluidDynamicViscosity = 0.05  # kg / (m s)
fluidDensity  = 100           # kg m^-3

fluidPRef = 0.0

fluidPInit = fluidPRef

# Set solver parameters
fsiDynamicSolverTheta    = [1.0]
nonlinearMaximumIterations      = 100000000 #default: 100000
nonlinearRelativeTolerance      = 1.0E-4    #default: 1.0E-05
nonlinearAbsoluteTolerance      = 1.0E-4    #default: 1.0E-10
nonlinearMaxFunctionEvaluations = 100000
nonlinearLinesearchAlpha        = 1.0
linearMaximumIterations      = 100000000 #default: 100000
linearRelativeTolerance      = 1.0E-4    #default: 1.0E-05
linearAbsoluteTolerance      = 1.0E-4    #default: 1.0E-10
linearDivergenceTolerance    = 1.0E5     #default: 1.0E5
linearRestartValue           = 30        #default: 30

#================================================================================================================================
#  Should not need to change anything below here.
#================================================================================================================================

highestInterpolation 
if (useHermite):
    numberOfNodesXi = 2
else:
    numberOfNodesXi = 3

numberOfFluidNodes = (numberOfFluidX1Elements*numberOfNodesXi)*(numberOfSolidYElements*(numberOfNodesXi-1))+ \
                    (numberOfFluidX2Elements*numberOfNodesXi)*(numberOfSolidYElements*(numberOfNodesXi-1))+ \
                    ((numberOfFluidX1Elements+numberOfFluidX2Elements+numberOfSolidXElements)*(numberOfNodesXi-1)+1)* \
                    (numberOfFluidYElements*(numberOfNodesXi-1)+1)
numberOfFluidElements = (numberOfFluidX1Elements+numberOfFluidX2Elements+numberOfSolidXElements)* \
                        (numberOfSolidYElements+numberOfFluidYElements) - numberOfSolidElements

contextUserNumber = 1

solidCoordinateSystemUserNumber     = 1
fluidCoordinateSystemUserNumber     = 2
interfaceCoordinateSystemUserNumber = 3
  
solidRegionUserNumber = 1
fluidRegionUserNumber = 2
interfaceUserNumber   = 3

linearBasisUserNumber = 1
quadraticBasisUserNumber = 2
hermiteBasisUserNumber = 3
interfaceQuadraticBasisUserNumber = 4
interfaceHermiteBasisUserNumber = 5

solidMeshUserNumber     = 1
fluidMeshUserNumber     = 2
interfaceMeshUserNumber = 3
movingMeshUserNumber    = 4
  
solidDecompositionUserNumber     = 1
fluidDecompositionUserNumber     = 2
interfaceDecompositionUserNumber = 3
  
solidGeometricFieldUserNumber     = 11
solidFibreFieldUserNumber     = 12
solidEquationsSetFieldUserNumber = 13
solidDependentFieldUserNumber = 14
solidMaterialsFieldUserNumber = 15
solidSourceFieldUserNumber = 16

fluidGeometricFieldUserNumber     = 21
fluidEquationsSetFieldUserNumber = 22
fluidDependentFieldUserNumber = 23
fluidMaterialsFieldUserNumber = 24
fluidIndependentFieldUserNumber = 25
bcCellMLModelsFieldUserNumber = 26
bcCellMLStateFieldUserNumber = 27
bcCellMLParametersFieldUserNumber = 28
bcCellMLIntermediateFieldUserNumber = 29

movingMeshEquationsSetFieldUserNumber = 31
movingMeshDependentFieldUserNumber    = 32
movingMeshMaterialsFieldUserNumber    = 33
movingMeshIndependentFieldUserNumber  = 34

interfaceGeometricFieldUserNumber = 41
interfaceLagrangeFieldUserNumber  = 42
 
solidEquationsSetUserNumber  = 1
fluidEquationsSetUserNumber  = 2
movingMeshEquationsSetUserNumber = 3

bcCellMLUserNumber = 1

interfaceConditionUserNumber = 1
  
fsiProblemUserNumber = 1
  
DynamicSolverIndex = 1
LinearSolverMovingMeshIndex = 2
  
IndependentFieldMovingMeshUserNumberK = 1
LinearSolverMovingMeshEquationsUserNumber = 122

SolidEquationsSetIndex  = 1
FluidEquationsSetIndex  = 2
InterfaceConditionIndex = 1
SolidMeshIndex = 1
FluidMeshIndex = 2

derivIdx   = 1
versionIdx = 1
    
#================================================================================================================================
#  Initialise OpenCMISS
#================================================================================================================================

# Import the libraries (OpenCMISS,python,numpy,scipy)
import numpy,csv,time,sys,os,pdb
from opencmiss.iron import iron

context = iron.Context()
context.Create(contextUserNumber)

# Diagnostics
iron.ErrorHandlingModeSet(iron.ErrorHandlingModes.TRAP_ERROR)

# Get the computational nodes info
computationEnvironment = iron.ComputationEnvironment()
numberOfComputationalNodes = computationEnvironment.NumberOfWorldNodesGet()
computationalNodeNumber = computationEnvironment.WorldNodeNumberGet()
        
#================================================================================================================================
#  Initial Data & Default Values
#================================================================================================================================

# (NONE/TIMING/MATRIX/ELEMENT_MATRIX/NODAL_MATRIX)
fluidEquationsSetOutputType = iron.EquationsSetOutputTypes.NONE
#fluidEquationsSetOutputType = iron.EquationsSetOutputTypes.PROGRESS
fluidEquationsOutputType = iron.EquationsOutputTypes.NONE
#fluidEquationsOutputType = iron.EquationsOutputTypes.TIMING
#fluidEquationsOutputType = iron.EquationsOutputTypes.MATRIX
#fluidEquationsOutputType = iron.EquationsOutputTypes.ELEMENT_MATRIX
#fsiDynamicSolverOutputType = iron.SolverOutputTypes.NONE
fsiDynamicSolverOutputType = iron.SolverOutputTypes.PROGRESS
#fsiDynamicSolverOutputType = iron.SolverOutputTypes.MATRIX
#fsiNonlinearSolverOutputType = iron.SolverOutputTypes.NONE
fsiNonlinearSolverOutputType = iron.SolverOutputTypes.PROGRESS
#fsiNonlinearSolverOutputType = iron.SolverOutputTypes.MATRIX
#fsiLinearSolverOutputType = iron.SolverOutputTypes.NONE
fsiLinearSolverOutputType = iron.SolverOutputTypes.PROGRESS
#fsiLinearSolverOutputType = iron.SolverOutputTypes.MATRIX

if (setupOutput):
    print('SUMMARY')
    print('=======')
    print(' ')
    print('  Temporal parameters')
    print('  -------------------')
    print(' ')
    print('  Start time:     %.3f s' % (startTime))
    print('  Stop time:      %.3f s' % (stopTime))
    print('  Time increment: %.5f s' % (timeStep))
    print(' ')
    print('  Material parameters')
    print('  -------------------')
    print(' ')
    print('    Fluid:')
    print('      Dynamic viscosity: {0:.3f} kg.m^-1.s^-1'.format(fluidDynamicViscosity))
    print('      Density: {0:.3f} kg.m^-3'.format(fluidDensity))
    print(' ')
    print('  Mesh parameters')
    print('  -------------------')
    print(' ')
    print('    Number of dimensions: {0:d}'.format(numberOfDimensions))
    print('    Use Hermite: {}'.format(useHermite))
    print('    Fluid:')
    print('      Number of X1 elements: {0:d}'.format(numberOfFluidX1Elements))
    print('      Number of X2 elements: {0:d}'.format(numberOfFluidX2Elements))
    print('      Number of Y  elements: {0:d}'.format(numberOfFluidYElements))
    print('      Number of nodes: {0:d}'.format(numberOfFluidNodes))
    print('      Number of elements: {0:d}'.format(numberOfFluidElements))
 
#================================================================================================================================
#  Coordinate Systems
#================================================================================================================================

if (progressDiagnostics):
    print(' ')
    print('Coordinate systems ...')

# Create a RC coordinate system for the fluid region
fluidCoordinateSystem = iron.CoordinateSystem()
fluidCoordinateSystem.CreateStart(fluidCoordinateSystemUserNumber)
fluidCoordinateSystem.DimensionSet(numberOfDimensions)
fluidCoordinateSystem.CreateFinish()

if (progressDiagnostics):
    print('Coordinate systems ... Done')
  
#================================================================================================================================
#  Regions
#================================================================================================================================

if (progressDiagnostics):
    print('Regions ...')

# Create a fluid region
fluidRegion = iron.Region()
fluidRegion.CreateStart(fluidRegionUserNumber,iron.WorldRegion)
fluidRegion.label = 'FluidRegion'
fluidRegion.coordinateSystem = fluidCoordinateSystem
fluidRegion.CreateFinish()
    
if (progressDiagnostics):
    print('Regions ... Done')

#================================================================================================================================
#  Bases
#================================================================================================================================

if (progressDiagnostics):
    print('Basis functions ...')

numberOfGaussXi =3
    
linearBasis = iron.Basis()
linearBasis.CreateStart(linearBasisUserNumber)
linearBasis.type = iron.BasisTypes.LAGRANGE_HERMITE_TP
linearBasis.numberOfXi = numberOfDimensions
linearBasis.interpolationXi = [iron.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*numberOfDimensions
linearBasis.quadratureNumberOfGaussXi = [numberOfGaussXi]*numberOfDimensions
linearBasis.CreateFinish()
if (useHermite):
    numberOfNodesXi = 2
    hermiteBasis = iron.Basis()
    hermiteBasis.CreateStart(hermiteBasisUserNumber)
    hermiteBasis.type = iron.BasisTypes.LAGRANGE_HERMITE_TP
    hermiteBasis.numberOfXi = numberOfDimensions
    hermiteBasis.interpolationXi = [iron.BasisInterpolationSpecifications.CUBIC_HERMITE]*numberOfDimensions
    hermiteBasis.quadratureNumberOfGaussXi = [numberOfGaussXi]*numberOfDimensions
    hermiteBasis.CreateFinish()
    if (problemType == FSI):
        interfaceHermiteBasis = iron.Basis()
        interfaceHermiteBasis.CreateStart(interfaceHermiteBasisUserNumber)
        interfaceHermiteBasis.type = iron.BasisTypes.LAGRANGE_HERMITE_TP
        interfaceHermiteBasis.numberOfXi = numberOfInterfaceDimensions
        interfaceHermiteBasis.interpolationXi = [iron.BasisInterpolationSpecifications.CUBIC_HERMITE]*numberOfInterfaceDimensions
        interfaceHermiteBasis.quadratureNumberOfGaussXi = [numberOfGaussXi]*numberOfInterfaceDimensions
        interfaceHermiteBasis.CreateFinish()    
else:
    numberOfNodesXi = 3
    quadraticBasis = iron.Basis()
    quadraticBasis.CreateStart(quadraticBasisUserNumber)
    quadraticBasis.type = iron.BasisTypes.LAGRANGE_HERMITE_TP
    quadraticBasis.numberOfXi = numberOfDimensions
    quadraticBasis.interpolationXi = [iron.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]*numberOfDimensions
    quadraticBasis.quadratureNumberOfGaussXi = [numberOfGaussXi]*numberOfDimensions
    quadraticBasis.CreateFinish()
    if (problemType == FSI):
        interfaceQuadraticBasis = iron.Basis()
        interfaceQuadraticBasis.CreateStart(interfaceQuadraticBasisUserNumber)
        interfaceQuadraticBasis.type = iron.BasisTypes.LAGRANGE_HERMITE_TP
        interfaceQuadraticBasis.numberOfXi = numberOfInterfaceDimensions
        interfaceQuadraticBasis.interpolationXi = [iron.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]*numberOfInterfaceDimensions
        interfaceQuadraticBasis.quadratureNumberOfGaussXi = [numberOfGaussXi]*numberOfInterfaceDimensions
        interfaceQuadraticBasis.CreateFinish()    

if (progressDiagnostics):
    print('Basis functions ... Done')
  
#================================================================================================================================
#  Mesh
#================================================================================================================================

if (progressDiagnostics):
    print('Meshes ...')    
                   
if (problemType != FLUID):
    solidNodes = iron.Nodes()
    solidNodes.CreateStart(solidRegion,numberOfSolidNodes)
    solidNodes.CreateFinish()

    solidMesh = iron.Mesh()
    solidMesh.CreateStart(solidMeshUserNumber,solidRegion,numberOfDimensions)
    solidMesh.NumberOfElementsSet(numberOfSolidElements)
    solidMesh.NumberOfComponentsSet(2)

    if (useHermite):
        solidHermiteElements = iron.MeshElements()
        solidHermiteElements.CreateStart(solidMesh,1,hermiteBasis)
    else:
        solidQuadraticElements = iron.MeshElements()
        solidQuadraticElements.CreateStart(solidMesh,1,quadraticBasis)
    solidLinearElements = iron.MeshElements()
    solidLinearElements.CreateStart(solidMesh,2,linearBasis)
        
    # Solid mesh elements
    if (debugLevel > 2):
        print('  Solid Elements:')
    for yElementIdx in range(1,numberOfSolidYElements+1):
        for xElementIdx in range(1,numberOfSolidXElements+1):
            elementNumber = xElementIdx+(yElementIdx-1)*numberOfSolidXElements
            localNodes1=(xElementIdx-1)*(numberOfNodesXi-1)+1+ \
                         (yElementIdx-1)*(numberOfSolidXElements*(numberOfNodesXi-1)+1)*(numberOfNodesXi-1)
            localNodes3=localNodes1+(numberOfNodesXi-1)
            localNodes7=localNodes1+(numberOfSolidXElements*(numberOfNodesXi-1)+1)*(numberOfNodesXi-1)
            localNodes9=localNodes7+(numberOfNodesXi-1)
            solidLinearElements.NodesSet(elementNumber,[localNodes1,localNodes3,localNodes7,localNodes9])
            if (useHermite):
                solidHermiteElements.NodesSet(elementNumber,[localNodes1,localNodes3,localNodes7,localNodes9])
                if (debugLevel > 2):
                    print('    Element %8d; Nodes: %8d, %8d, %8d, %8d' % \
                          (elementNumber,localNodes1,localNodes3,localNodes7,localNodes9))
            else:
                localNodes2=localNodes1+1
                localNodes4=localNodes1+(numberOfSolidXElements*(numberOfNodesXi-1)+1)
                localNodes5=localNodes4+1
                localNodes6=localNodes5+1
                localNodes8=localNodes7+1
                solidQuadraticElements.NodesSet(elementNumber,[localNodes1,localNodes2,localNodes3,localNodes4, \
                                                               localNodes5,localNodes6,localNodes7,localNodes8,localNodes9])
                if (debugLevel > 2):
                    print('    Element %8d; Nodes: %8d, %8d, %8d, %8d, %8d, %8d, %8d, %8d, %8d' % \
                          (elementNumber,localNodes1,localNodes2,localNodes3,localNodes4,localNodes5,\
                           localNodes6,localNodes7,localNodes8,localNodes9))

    if (useHermite):
        solidHermiteElements.CreateFinish()
    else:
        solidQuadraticElements.CreateFinish()
    solidLinearElements.CreateFinish()

    solidMesh.CreateFinish()

if (problemType != SOLID):
    fluidNodes = iron.Nodes()
    fluidNodes.CreateStart(fluidRegion,numberOfFluidNodes)
    fluidNodes.CreateFinish()

    fluidMesh = iron.Mesh()
    fluidMesh.CreateStart(fluidMeshUserNumber,fluidRegion,numberOfDimensions)
    fluidMesh.NumberOfElementsSet(numberOfFluidElements)
    fluidMesh.NumberOfComponentsSet(2)

    if (useHermite):
        fluidHermiteElements = iron.MeshElements()
        fluidHermiteElements.CreateStart(fluidMesh,1,hermiteBasis)
    else:
        fluidQuadraticElements = iron.MeshElements()
        fluidQuadraticElements.CreateStart(fluidMesh,1,quadraticBasis)
    fluidLinearElements = iron.MeshElements()
    fluidLinearElements.CreateStart(fluidMesh,2,linearBasis)
        
    # Fluid mesh elements
    if (debugLevel > 2):
        print('  Fluid Elements:')
    for yElementIdx in range(1,numberOfSolidYElements+1):
        # Elements to the left of the solid
        for xElementIdx in range(1,numberOfFluidX1Elements+1):
            elementNumber = xElementIdx+(yElementIdx-1)*(numberOfFluidX1Elements+numberOfFluidX2Elements)
            localNodes1= (xElementIdx-1)*(numberOfNodesXi-1)+1+\
                         (yElementIdx-1)*((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)*(numberOfNodesXi-1)
            localNodes3=localNodes1+(numberOfNodesXi-1)
            localNodes7=localNodes1+((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)* \
                         (numberOfNodesXi-1)
            localNodes9=localNodes7+(numberOfNodesXi-1)
            fluidLinearElements.NodesSet(elementNumber,[localNodes1,localNodes3,localNodes7,localNodes9])
            if (useHermite):
                fluidHermiteElements.NodesSet(elementNumber,[localNodes1,localNodes3,localNodes7,localNodes9])
                if (debugLevel > 2):
                    print('    Element %8d; Nodes: %8d, %8d, %8d, %8d' % \
                          (elementNumber,localNodes1,localNodes3,localNodes7,localNodes9))
            else:
                localNodes2=localNodes1+1
                localNodes4=localNodes1+((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)
                localNodes5=localNodes4+1
                localNodes6=localNodes5+1
                localNodes8=localNodes7+1
                fluidQuadraticElements.NodesSet(elementNumber,[localNodes1,localNodes2,localNodes3,localNodes4, \
                                                               localNodes5,localNodes6,localNodes7,localNodes8,localNodes9])
                if (debugLevel > 2):
                    print('    Element %8d; Nodes: %8d, %8d, %8d, %8d, %8d, %8d, %8d, %8d, %8d' % \
                          (elementNumber,localNodes1,localNodes2,localNodes3,localNodes4,localNodes5,\
                           localNodes6,localNodes7,localNodes8,localNodes9))
        # Elements to the right of the solid
        if(yElementIdx == numberOfSolidYElements):
            offset = numberOfSolidXElements*(numberOfNodesXi-1) - 1
        else:
            offset = 0
        for xElementIdx in range(1,numberOfFluidX2Elements+1):
            elementNumber = numberOfFluidX1Elements+xElementIdx+(yElementIdx-1)*(numberOfFluidX1Elements+numberOfFluidX2Elements)
            localNodes1=(numberOfFluidX1Elements*(numberOfNodesXi-1)+1)+\
                         (xElementIdx-1)*(numberOfNodesXi-1)+1+\
                         (yElementIdx-1)*((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)*(numberOfNodesXi-1)
            localNodes3=localNodes1+(numberOfNodesXi-1)
            localNodes7=localNodes1+\
                        ((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)*(numberOfNodesXi-1)+offset
            localNodes9=localNodes7+(numberOfNodesXi-1)
            fluidLinearElements.NodesSet(elementNumber,[localNodes1,localNodes3,localNodes7,localNodes9])
            if (useHermite):
                fluidHermiteElements.NodesSet(elementNumber,[localNodes1,localNodes3,localNodes7,localNodes9])
                if (debugLevel > 2):
                    print('    Element %8d; Nodes: %8d, %8d, %8d, %8d' % \
                          (elementNumber,localNodes1,localNodes3,localNodes7,localNodes9))
            else:
                localNodes2=localNodes1+1
                localNodes4=localNodes1+((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)
                localNodes5=localNodes4+1
                localNodes6=localNodes5+1
                localNodes8=localNodes7+1
                fluidQuadraticElements.NodesSet(elementNumber,[localNodes1,localNodes2,localNodes3,localNodes4, \
                                                               localNodes5,localNodes6,localNodes7,localNodes8,localNodes9])
                if (debugLevel > 2):
                    print('    Element %8d; Nodes: %8d, %8d, %8d, %8d, %8d, %8d, %8d, %8d, %8d' % \
                          (elementNumber,localNodes1,localNodes2,localNodes3,localNodes4,localNodes5,\
                           localNodes6,localNodes7,localNodes8,localNodes9))
    # Elements above the solid
    for yElementIdx in range(1,numberOfFluidYElements+1):
        for xElementIdx in range(1,numberOfFluidX1Elements+numberOfFluidX2Elements+numberOfSolidXElements+1):
            elementNumber = (numberOfFluidX1Elements+numberOfFluidX2Elements)*numberOfSolidYElements+xElementIdx+(yElementIdx-1)* \
                            (numberOfFluidX1Elements+numberOfFluidX2Elements+numberOfSolidXElements)
            localNodes1=((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)*\
                         (numberOfSolidYElements*(numberOfNodesXi-1))+ \
                         (xElementIdx-1)*(numberOfNodesXi-1)+1+(yElementIdx-1)*(numberOfNodesXi-1)* \
                         ((numberOfFluidX1Elements+numberOfFluidX2Elements+numberOfSolidXElements)*(numberOfNodesXi-1)+1)
            localNodes3=localNodes1+(numberOfNodesXi-1)
            localNodes7=localNodes1+((numberOfFluidX1Elements+numberOfFluidX2Elements+numberOfSolidXElements)* \
                                     (numberOfNodesXi-1)+1)*(numberOfNodesXi-1)
            localNodes9=localNodes7+(numberOfNodesXi-1)
            fluidLinearElements.NodesSet(elementNumber,[localNodes1,localNodes3,localNodes7,localNodes9])
            if (useHermite):
                fluidHermiteElements.NodesSet(elementNumber,[localNodes1,localNodes3,localNodes7,localNodes9])
                if (debugLevel > 2):
                    print('    Element %8d; Nodes: %8d, %8d, %8d, %8d' % \
                          (elementNumber,localNodes1,localNodes3,localNodes7,localNodes9))
            else:
                localNodes2=localNodes1+1
                localNodes4=localNodes1+(numberOfFluidX1Elements+numberOfFluidX2Elements+numberOfSolidXElements)*(numberOfNodesXi-1)+1
                localNodes5=localNodes4+1
                localNodes6=localNodes5+1
                localNodes8=localNodes7+1
                fluidQuadraticElements.NodesSet(elementNumber,[localNodes1,localNodes2,localNodes3,localNodes4, \
                                                               localNodes5,localNodes6,localNodes7,localNodes8,localNodes9])
                if (debugLevel > 2):
                    print('    Element %8d; Nodes: %8d, %8d, %8d, %8d, %8d, %8d, %8d, %8d, %8d' % \
                          (elementNumber,localNodes1,localNodes2,localNodes3,localNodes4,localNodes5,\
                           localNodes6,localNodes7,localNodes8,localNodes9))

    if (useHermite):
        fluidHermiteElements.CreateFinish()
    else:
        fluidQuadraticElements.CreateFinish()
    fluidLinearElements.CreateFinish()

    fluidMesh.CreateFinish()

if (progressDiagnostics):
    print('Meshes ... Done')    

#================================================================================================================================
#  Interface
#================================================================================================================================

if (problemType == FSI):
    if (progressDiagnostics):
        print('Interface ...')
    
        # Create an interface between the two meshes
        interface = iron.Interface()
        interface.CreateStart(interfaceUserNumber,iron.WorldRegion)
        interface.LabelSet('Interface')
        # Add in the two meshes
        solidMeshIndex = interface.MeshAdd(solidMesh)
        fluidMeshIndex = interface.MeshAdd(fluidMesh)
        interface.CoordinateSystemSet(interfaceCoordinateSystem)
        interface.CreateFinish()
        
    if (progressDiagnostics):
        print('Interface ... Done')
            
#================================================================================================================================
#  Interface Mesh
#================================================================================================================================

if (problemType == FSI):
    if (progressDiagnostics):
        print('Interface Mesh ...')
    
    # Create an interface mesh
    InterfaceNodes = iron.Nodes()
    InterfaceNodes.CreateStartInterface(interface,numberOfInterfaceNodes)
    InterfaceNodes.CreateFinish()
    
    interfaceMesh = iron.Mesh()
    interfaceMesh.CreateStartInterface(interfaceMeshUserNumber,interface,numberOfInterfaceDimensions)
    interfaceMesh.NumberOfElementsSet(numberOfInterfaceElements)
    interfaceMesh.NumberOfComponentsSet(1)
    
    if (useHermite):
        interfaceHermiteElements = iron.MeshElements()
        interfaceHermiteElements.CreateStart(interfaceMesh,1,interfaceHermiteBasis)
    else:
        interfaceQuadraticElements = iron.MeshElements()
        interfaceQuadraticElements.CreateStart(interfaceMesh,1,interfaceQuadraticBasis)
        
    if (debugLevel > 2):
        print('  Interface Elements:')
    elementNumber = 0
    for interfaceElementIdx in range(1,numberOfSolidXElements + 2*numberOfSolidYElements + 1):
        elementNumber = elementNumber + 1
        localNodes1 = (interfaceElementIdx-1)*(numberOfNodesXi-1)+1
        localNodes3 = localNodes1 + (numberOfNodesXi-1)
        if (useHermite):
            interfaceHermiteElements.NodesSet(elementNumber,[localNodes1,localNodes3])
            if (debugLevel > 2):
                print('    Element %8d; Nodes: %8d, %8d' % (elementNumber,localNodes1,localNodes3))
        else:
            localNodes2 = localNodes1 + 1
            interfaceQuadraticElements.NodesSet(elementNumber,[localNodes1,localNodes2,localNodes3])
            if (debugLevel > 2):
                print('    Element %8d; Nodes: %8d, %8d, %8d' % (elementNumber,localNodes1,localNodes2,localNodes3))

    if (useHermite):
        interfaceHermiteElements.CreateFinish()
    else:
        interfaceQuadraticElements.CreateFinish()

    interfaceMesh.CreateFinish()

    if (progressDiagnostics):
        print('Interface Mesh ... Done')
    

#================================================================================================================================
#  Mesh Connectivity
#================================================================================================================================

if (problemType == FSI):
    if (progressDiagnostics):
        print('Interface Mesh Connectivity ...')

    # Couple the interface meshes
    interfaceMeshConnectivity = iron.InterfaceMeshConnectivity()
    interfaceMeshConnectivity.CreateStart(interface,interfaceMesh)
    if (useHermite):
        interfaceMeshConnectivity.BasisSet(interfaceHermiteBasis)
    else:
        interfaceMeshConnectivity.BasisSet(interfaceQuadraticBasis)
        
    interfaceElementNumber = 0
    interfaceNodes = [0]*(numberOfInterfaceNodes)
    solidNodes = [0]*(numberOfInterfaceNodes)
    fluidNodes = [0]*(numberOfInterfaceNodes)
    localInterfaceNodes = [0]*numberOfNodesXi
    localSolidNodes = [0]*numberOfNodesXi
    localFluidNodes = [0]*numberOfNodesXi
    # Left edge of solid
    for interfaceElementIdx in range(1,numberOfSolidYElements+1):
        interfaceElementNumber = interfaceElementNumber + 1
        if (debugLevel > 2):
            print('  Interface Element %8d:' % (interfaceElementNumber))        
        solidElementNumber = (interfaceElementIdx - 1)*numberOfSolidXElements + 1
        fluidElementNumber = numberOfFluidX1Elements+(interfaceElementIdx - 1)*(numberOfFluidX1Elements + numberOfFluidX2Elements)
        # Map interface elements
        interfaceMeshConnectivity.ElementNumberSet(interfaceElementNumber,solidMeshIndex,solidElementNumber)
        interfaceMeshConnectivity.ElementNumberSet(interfaceElementNumber,fluidMeshIndex,fluidElementNumber)
        if (debugLevel > 2):
            print('    Solid Element %8d; Fluid Element %8d' % (solidElementNumber,fluidElementNumber))        
        localInterfaceNodes[0] = (interfaceElementIdx - 1)*(numberOfNodesXi - 1) + 1
        localSolidNodes[0] = (interfaceElementIdx - 1)*(numberOfNodesXi - 1)*(numberOfSolidXElements*(numberOfNodesXi - 1) + 1)+1
        localFluidNodes[0] = numberOfFluidX1Elements*(numberOfNodesXi-1) + 1 + \
                             (interfaceElementIdx - 1)*(numberOfNodesXi - 1)*\
                             ((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi - 1) + 2)
        if (not useHermite):
            localInterfaceNodes[1] = localInterfaceNodes[0]+1
            localSolidNodes[1] = localSolidNodes[0] + numberOfSolidXElements*(numberOfNodesXi-1)+1
            localFluidNodes[1] = localFluidNodes[0] + ((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)
        localInterfaceNodes[numberOfNodesXi-1] = localInterfaceNodes[0] + (numberOfNodesXi - 1)
        localSolidNodes[numberOfNodesXi-1] = localSolidNodes[0] + \
                                             (numberOfNodesXi - 1)*(numberOfSolidXElements*(numberOfNodesXi - 1) + 1)
        localFluidNodes[numberOfNodesXi-1] = localFluidNodes[0] + ((numberOfFluidX1Elements+numberOfFluidX2Elements)*\
                                                                   (numberOfNodesXi-1)+2)*(numberOfNodesXi - 1)
        # Map interface xi
        for localNodeIdx in range(0,numberOfNodesXi):
            xi=float(localNodeIdx)/float(numberOfNodesXi-1)
            solidXi = [0.0,xi]
            fluidXi = [1.0,xi]
            interfaceNodes[localInterfaceNodes[localNodeIdx]-1]=localInterfaceNodes[localNodeIdx]
            solidNodes[localInterfaceNodes[localNodeIdx]-1]=localSolidNodes[localNodeIdx]
            fluidNodes[localInterfaceNodes[localNodeIdx]-1]=localFluidNodes[localNodeIdx]
            interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,solidMeshIndex,solidElementNumber,localNodeIdx+1,1,solidXi)
            interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,fluidMeshIndex,fluidElementNumber,localNodeIdx+1,1,fluidXi)
            if (debugLevel > 2):
                print('    Local node    %8d:' % (localNodeIdx+1))        
                print('      Interface node    %8d:' % (localInterfaceNodes[localNodeIdx]))        
                print('      Solid node        %8d; Solid xi = [%.2f, %.2f ]' % (localSolidNodes[localNodeIdx],solidXi[0],solidXi[1]))
                print('      Fluid node        %8d; Fluid xi = [%.2f, %.2f ]' % (localFluidNodes[localNodeIdx],fluidXi[0],fluidXi[1]))
    # Top edge of solid
    for interfaceElementIdx in range(1,numberOfSolidXElements+1):
        interfaceElementNumber = interfaceElementNumber + 1
        if (debugLevel > 2):
            print('  Interface Element %8d:' % (interfaceElementNumber))        
        solidElementNumber = interfaceElementIdx + numberOfSolidXElements*(numberOfSolidYElements - 1)
        fluidElementNumber = interfaceElementIdx + numberOfFluidX1Elements + \
                             (numberOfFluidX1Elements+numberOfFluidX2Elements)*numberOfSolidYElements
        # Map interface elements
        interfaceMeshConnectivity.ElementNumberSet(interfaceElementNumber,solidMeshIndex,solidElementNumber)
        interfaceMeshConnectivity.ElementNumberSet(interfaceElementNumber,fluidMeshIndex,fluidElementNumber)
        if (debugLevel > 2):
            print('    Solid Element %8d; Fluid Element %8d' % (solidElementNumber,fluidElementNumber))        
        localInterfaceNodes[0] = (interfaceElementIdx - 1)*(numberOfNodesXi - 1) + 1 + (numberOfSolidYElements)*(numberOfNodesXi-1)
        localSolidNodes[0] = (interfaceElementIdx - 1)*(numberOfNodesXi - 1) + 1 + \
                                 (numberOfSolidXElements*(numberOfNodesXi-1)+1)*(numberOfSolidYElements)*(numberOfNodesXi-1)        
        localFluidNodes[0] = (interfaceElementIdx - 1)*(numberOfNodesXi - 1) + 1 + numberOfFluidX1Elements*(numberOfNodesXi-1) + \
                             ((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi - 1) + 2)*\
                             numberOfSolidYElements*(numberOfNodesXi-1)
        if (not useHermite):
            localInterfaceNodes[1] = localInterfaceNodes[0]+1
            localSolidNodes[1] = localSolidNodes[0] + 1
            localFluidNodes[1] = localFluidNodes[0] + 1
        localInterfaceNodes[numberOfNodesXi-1] = localInterfaceNodes[0] + (numberOfNodesXi - 1)
        localSolidNodes[numberOfNodesXi-1] = localSolidNodes[0] + (numberOfNodesXi - 1)
        localFluidNodes[numberOfNodesXi-1] = localFluidNodes[0] + (numberOfNodesXi - 1)        
        # Map interface xi
        for localNodeIdx in range(0,numberOfNodesXi):
            xi=float(localNodeIdx)/float(numberOfNodesXi-1)
            solidXi = [xi,1.0]
            fluidXi = [xi,0.0]
            interfaceNodes[localInterfaceNodes[localNodeIdx]-1]=localInterfaceNodes[localNodeIdx]
            solidNodes[localInterfaceNodes[localNodeIdx]-1]=localSolidNodes[localNodeIdx]
            fluidNodes[localInterfaceNodes[localNodeIdx]-1]=localFluidNodes[localNodeIdx]
            interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,solidMeshIndex,solidElementNumber,localNodeIdx+1,1,solidXi)
            interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,fluidMeshIndex,fluidElementNumber,localNodeIdx+1,1,fluidXi)
            if (debugLevel > 2):
                print('    Local node    %8d:' % (localNodeIdx+1))        
                print('      Interface node    %8d:' % (localInterfaceNodes[localNodeIdx]))        
                print('      Solid node        %8d; Solid xi = [%.2f, %.2f ]' % (localSolidNodes[localNodeIdx],solidXi[0],solidXi[1]))
                print('      Fluid node        %8d; Fluid xi = [%.2f, %.2f ]' % (localFluidNodes[localNodeIdx],fluidXi[0],fluidXi[1]))
    # right edge of solid
    for interfaceElementIdx in range(1,numberOfSolidYElements+1):
        if (interfaceElementIdx == 1):
            offset = numberOfSolidXElements*(numberOfNodesXi-1) 
        else:
            offset = 1
        interfaceElementNumber = interfaceElementNumber + 1
        if (debugLevel > 2):
            print('  Interface Element %8d:' % (interfaceElementNumber))
        solidElementNumber = (numberOfSolidYElements - interfaceElementIdx + 1)*numberOfSolidXElements 
        fluidElementNumber = (numberOfSolidYElements - interfaceElementIdx)*(numberOfFluidX1Elements + numberOfFluidX2Elements) \
                             + numberOfFluidX1Elements + 1
        # Map interface elements
        interfaceMeshConnectivity.ElementNumberSet(interfaceElementNumber,solidMeshIndex,solidElementNumber)
        interfaceMeshConnectivity.ElementNumberSet(interfaceElementNumber,fluidMeshIndex,fluidElementNumber)
        if (debugLevel > 2):
            print('    Solid Element %8d; Fluid Element %8d' % (solidElementNumber,fluidElementNumber))        
        localInterfaceNodes[0] = (interfaceElementIdx - 1)*(numberOfNodesXi - 1) + \
                                 (numberOfSolidXElements + numberOfSolidYElements)*(numberOfNodesXi - 1) + 1
        localSolidNodes[0] = (numberOfSolidXElements*(numberOfNodesXi - 1) + 1)* \
                             ((numberOfSolidYElements - interfaceElementIdx + 1)*(numberOfNodesXi-1)+1)
        localFluidNodes[0] = ((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi - 1) + 2)*\
                             (numberOfSolidYElements - interfaceElementIdx + 1)*(numberOfNodesXi - 1) + \
                             numberOfFluidX1Elements*(numberOfNodesXi-1)+ 1 + offset
        if (not useHermite):
            localInterfaceNodes[1] = localInterfaceNodes[0]+1
            localSolidNodes[1] = localSolidNodes[0] - numberOfSolidXElements*(numberOfNodesXi-1) - 1
            localFluidNodes[1] = localFluidNodes[0] - ((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2) \
                                 - offset + 1
        localInterfaceNodes[numberOfNodesXi-1] = localInterfaceNodes[0] + numberOfNodesXi - 1
        localSolidNodes[numberOfNodesXi-1] = localSolidNodes[0] - \
                                             (numberOfNodesXi - 1)*(numberOfSolidXElements*(numberOfNodesXi - 1) + 1)
        localFluidNodes[numberOfNodesXi-1] = localFluidNodes[0] - \
                                             ((numberOfFluidX1Elements+numberOfFluidX2Elements)*\
                                              (numberOfNodesXi-1)+2)*(numberOfNodesXi - 1) -offset+1
        # Map interface xi
        for localNodeIdx in range(0,numberOfNodesXi):
            xi=float(numberOfNodesXi-localNodeIdx-1)/float(numberOfNodesXi-1)
            solidXi = [1.0,xi]
            fluidXi = [0.0,xi]
            interfaceNodes[localInterfaceNodes[localNodeIdx]-1]=localInterfaceNodes[localNodeIdx]
            solidNodes[localInterfaceNodes[localNodeIdx]-1]=localSolidNodes[localNodeIdx]
            fluidNodes[localInterfaceNodes[localNodeIdx]-1]=localFluidNodes[localNodeIdx]
            interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,solidMeshIndex,solidElementNumber,localNodeIdx+1,1,solidXi)
            interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,fluidMeshIndex,fluidElementNumber,localNodeIdx+1,1,fluidXi)
            if (debugLevel > 2):
                print('    Local node    %8d:' % (localNodeIdx+1))        
                print('      Interface node    %8d:' % (localInterfaceNodes[localNodeIdx]))        
                print('      Solid node        %8d; Solid xi = [%.2f, %.2f ]' % (localSolidNodes[localNodeIdx],solidXi[0],solidXi[1]))
                print('      Fluid node        %8d; Fluid xi = [%.2f, %.2f ]' % (localFluidNodes[localNodeIdx],fluidXi[0],fluidXi[1]))
    # Map interface nodes
    interfaceMeshConnectivity.NodeNumberSet(interfaceNodes,solidMeshIndex,solidNodes,fluidMeshIndex,fluidNodes)        

    interfaceMeshConnectivity.CreateFinish()

    if (progressDiagnostics):
        print('Interface Mesh Connectivity ... Done')

#================================================================================================================================
#  Decomposition
#================================================================================================================================

if (progressDiagnostics):
    print('Decomposition ...')
    
if (problemType != FLUID):
    # Create a decomposition for the solid mesh
    solidDecomposition = iron.Decomposition()
    solidDecomposition.CreateStart(solidDecompositionUserNumber,solidMesh)
    solidDecomposition.TypeSet(iron.DecompositionTypes.CALCULATED)
    solidDecomposition.NumberOfDomainsSet(numberOfComputationalNodes)
    solidDecomposition.CalculateFacesSet(True)
    solidDecomposition.CreateFinish()

if (problemType != SOLID):
    # Create a decomposition for the fluid mesh
    fluidDecomposition = iron.Decomposition()
    fluidDecomposition.CreateStart(fluidDecompositionUserNumber,fluidMesh)
    fluidDecomposition.TypeSet(iron.DecompositionTypes.CALCULATED)
    fluidDecomposition.NumberOfDomainsSet(numberOfComputationalNodes)
    fluidDecomposition.CalculateFacesSet(True)
    fluidDecomposition.CreateFinish()

if (problemType == FSI):
    # Create a decomposition for the interface mesh
    interfaceDecomposition = iron.Decomposition()
    interfaceDecomposition.CreateStart(interfaceDecompositionUserNumber,interfaceMesh)
    interfaceDecomposition.TypeSet(iron.DecompositionTypes.CALCULATED)
    interfaceDecomposition.NumberOfDomainsSet(numberOfComputationalNodes)
    interfaceDecomposition.CreateFinish()

if (progressDiagnostics):
    print('Decomposition ... Done')
    
#================================================================================================================================
#  Geometric Field
#================================================================================================================================

if (progressDiagnostics):
    print('Geometric Field ...')

if (problemType != FLUID):    
    # Start to create a default (geometric) field on the solid region
    solidGeometricField = iron.Field()
    solidGeometricField.CreateStart(solidGeometricFieldUserNumber,solidRegion)
    # Set the decomposition to use
    solidGeometricField.DecompositionSet(solidDecomposition)
    # Set the scaling to use
    if (useHermite):
        solidGeometricField.ScalingTypeSet(iron.FieldScalingTypes.ARITHMETIC_MEAN)
    else:
        solidGeometricField.ScalingTypeSet(iron.FieldScalingTypes.NONE)
        solidGeometricField.VariableLabelSet(iron.FieldVariableTypes.U,'SolidGeometry')
    # Set the domain to be used by the field components.
    solidGeometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,1,1)
    solidGeometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,2,1)
    # Finish creating the first field
    solidGeometricField.CreateFinish()

if (problemType != SOLID):
    # Start to create a default (geometric) field on the fluid region
    fluidGeometricField = iron.Field()
    fluidGeometricField.CreateStart(fluidGeometricFieldUserNumber,fluidRegion)
    # Set the decomposition to use
    if (useHermite):
        fluidGeometricField.ScalingTypeSet(iron.FieldScalingTypes.ARITHMETIC_MEAN)
    else:
        fluidGeometricField.DecompositionSet(fluidDecomposition)
    # Set the scaling to use
    fluidGeometricField.ScalingTypeSet(iron.FieldScalingTypes.NONE)
    fluidGeometricField.VariableLabelSet(iron.FieldVariableTypes.U,'FluidGeometry')
    # Set the domain to be used by the field components.
    fluidGeometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,1,1)
    fluidGeometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,2,1)
    # Finish creating the second field
    fluidGeometricField.CreateFinish()

if (problemType == FSI):
    # Start to create a default (geometric) field on the Interface
    interfaceGeometricField = iron.Field()
    interfaceGeometricField.CreateStartInterface(interfaceGeometricFieldUserNumber,interface)
    # Set the decomposition to use
    interfaceGeometricField.DecompositionSet(interfaceDecomposition)
    # Set the scaling to use
    if (useHermite):
        interfaceGeometricField.ScalingTypeSet(iron.FieldScalingTypes.ARITHMETIC_MEAN)
    else:
        interfaceGeometricField.ScalingTypeSet(iron.FieldScalingTypes.NONE)
    interfaceGeometricField.VariableLabelSet(iron.FieldVariableTypes.U,'InterfaceGeometry')
    # Set the domain to be used by the field components.
    interfaceGeometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,1,1)
    interfaceGeometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,2,1)
    # Finish creating the first field
    interfaceGeometricField.CreateFinish()

if (progressDiagnostics):
    print('Geometric Field ... Done')
    
if (progressDiagnostics):
    print('Geometric Parameters ...')
    
if (problemType != FLUID):
    # Solid nodes
    if (debugLevel > 2):
        print('  Solid Nodes:')
    for yNodeIdx in range(1,numberOfSolidYElements*(numberOfNodesXi-1)+2):
        for xNodeIdx in range(1,numberOfSolidXElements*(numberOfNodesXi-1)+2):
            nodeNumber = xNodeIdx+(yNodeIdx-1)*(numberOfSolidXElements*(numberOfNodesXi-1)+1)
            nodeDomain = solidDecomposition.NodeDomainGet(nodeNumber,1)
            if (nodeDomain == computationalNodeNumber):
                xPosition = fluidX1Size + float(xNodeIdx-1)/float(numberOfSolidXElements*(numberOfNodesXi-1))*solidXSize
                yPosition = float(yNodeIdx-1)/float(numberOfSolidYElements*(numberOfNodesXi-1))*solidYSize
                solidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                             1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,xPosition)
                solidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                             1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,yPosition)
                if (debugLevel > 2):
                    print('      Node        %d:' % (nodeNumber))
                    print('         Position         = [ %.2f, %.2f ]' % (xPosition,yPosition))                 
                if (useHermite):
                    solidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,1,1.0)
                    solidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,2,0.0)
                    solidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,1,0.0)
                    solidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,2,1.0)
                    solidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,1,0.0)
                    solidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,2,0.0)
                    if (debugLevel > 2):
                        print('        S1 derivative    = [ %.2f, %.2f ]' % (1.0,0.0))                 
                        print('        S2 derivative    = [ %.2f, %.2f ]' % (0.0,1.0))                 
                        print('        S1xS2 derivative = [ %.2f, %.2f ]' % (0.0,0.0))
    # Update fields            
    solidGeometricField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)
    solidGeometricField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)

if (problemType != SOLID):                        
    if (debugLevel > 2):
        print('  Fluid Nodes:')
    for yNodeIdx in range(1,numberOfSolidYElements*(numberOfNodesXi-1)+1):
        # Nodes to the left of the solid
        for xNodeIdx in range(1,numberOfFluidX1Elements*(numberOfNodesXi-1)+2):
            nodeNumber = xNodeIdx+(yNodeIdx-1)*((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)
            nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,1)
            if (nodeDomain == computationalNodeNumber):
                xPosition = float(xNodeIdx-1)/float(numberOfFluidX1Elements*(numberOfNodesXi-1))*fluidX1Size
                yPosition = float(yNodeIdx-1)/float(numberOfSolidYElements*(numberOfNodesXi-1))*solidYSize
                fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                         1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,xPosition)
                fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                         1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,yPosition)
                if (debugLevel > 2):
                    print('      Node        %d:' % (nodeNumber))
                    print('         Position         = [ %.2f, %.2f ]' % (xPosition,yPosition))                 
                if (useHermite):
                    fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,1,1.0)
                    fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,2,0.0)
                    fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,1,0.0)
                    fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,2,1.0)
                    fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,1,0.0)
                    fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,2,0.0)
                    if (debugLevel > 2):
                        print('        S1 derivative    = [ %.2f, %.2f ]' % (1.0,0.0))                 
                        print('        S2 derivative    = [ %.2f, %.2f ]' % (0.0,1.0))                 
                        print('        S1xS2 derivative = [ %.2f, %.2f ]' % (0.0,0.0))                                 
        # Nodes to the right of the solid
        for xNodeIdx in range(1,numberOfFluidX2Elements*(numberOfNodesXi-1)+2):
            nodeNumber = xNodeIdx+numberOfFluidX1Elements*(numberOfNodesXi-1)+1+\
                         (yNodeIdx-1)*((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)
            nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,1)
            if (nodeDomain == computationalNodeNumber):
                xPosition = fluidX1Size+solidXSize+float(xNodeIdx-1)/float(numberOfFluidX1Elements*(numberOfNodesXi-1))*fluidX1Size
                yPosition = float(yNodeIdx-1)/float(numberOfSolidYElements*(numberOfNodesXi-1))*solidYSize
                fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                             1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,xPosition)
                fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                             1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,yPosition)
                if (debugLevel > 2):
                    print('      Node        %d:' % (nodeNumber))
                    print('         Position         = [ %.2f, %.2f ]' % (xPosition,yPosition))                 
                if (useHermite):
                    fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,1,1.0)
                    fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,2,0.0)
                    fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,1,0.0)
                    fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,2,1.0)
                    fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,1,0.0)
                    fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,2,0.0)
                    if (debugLevel > 2):
                        print('        S1 derivative    = [ %.2f, %.2f ]' % (1.0,0.0))                 
                        print('        S2 derivative    = [ %.2f, %.2f ]' % (0.0,1.0))                 
                        print('        S1xS2 derivative = [ %.2f, %.2f ]' % (0.0,0.0))                                 
    # Nodes to the top of the solid
    for yNodeIdx in range(1,numberOfFluidYElements*(numberOfNodesXi-1)+2):
        for xNodeIdx in range(1,(numberOfFluidX1Elements+numberOfSolidXElements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2):
            nodeNumber = ((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)*\
                         numberOfSolidYElements*(numberOfNodesXi-1)+xNodeIdx+(yNodeIdx-1)*\
                         ((numberOfFluidX1Elements+numberOfSolidXElements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+1)
            nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,1)
            if (nodeDomain == computationalNodeNumber):
                xPosition = float(xNodeIdx-1)/float((numberOfFluidX1Elements+numberOfSolidXElements+numberOfFluidX2Elements)*\
                                                    (numberOfNodesXi-1))*(fluidX1Size+solidXSize+fluidX2Size)
                yPosition = float(yNodeIdx-1)/float(numberOfFluidYElements*(numberOfNodesXi-1))*fluidYSize+solidYSize
                fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                             1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,xPosition)
                fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                             1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,yPosition)
                if (debugLevel > 2):
                    print('      Node        %d:' % (nodeNumber))
                    print('         Position         = [ %.2f, %.2f ]' % (xPosition,yPosition))                 
                    if (useHermite):
                        fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                     1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,1,1.0)
                        fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                     1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,2,0.0)
                        fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                     1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,1,0.0)
                        fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                     1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,2,1.0)
                        fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                     1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,1,0.0)
                        fluidGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                     1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,2,0.0)
                        if (debugLevel > 2):
                            print('        S1 derivative    = [ %.2f, %.2f ]' % (1.0,0.0))                 
                            print('        S2 derivative    = [ %.2f, %.2f ]' % (0.0,1.0))                 
                            print('        S1xS2 derivative = [ %.2f, %.2f ]' % (0.0,0.0))                                 
    # Update fields            
    fluidGeometricField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)
    fluidGeometricField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)

if (problemType == FSI):
    if (debugLevel > 2):
        print('  Interface Nodes:')
    # Left edge of interface nodes    
    for yNodeIdx in range(1,numberOfSolidYElements*(numberOfNodesXi-1)+1):
        nodeNumber = yNodeIdx
        #nodeDomain = interfaceDecomposition.NodeDomainGet(nodeNumber,1)
        nodeDomain = computationalNodeNumber
        if (nodeDomain == computationalNodeNumber):
            xPosition = fluidX1Size
            yPosition = float(yNodeIdx-1)/float(numberOfSolidYElements*(numberOfNodesXi-1))*solidYSize
            interfaceGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                             1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,xPosition)
            interfaceGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                             1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,yPosition)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
                print('         Position         = [ %.2f, %.2f ]' % (xPosition,yPosition))                 
            if (useHermite):
                interfaceGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,1,0.0)
                interfaceGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,2,1.0)
                if (debugLevel > 2):
                    print('        S1 derivative    = [ %.2f, %.2f ]' % (1.0,0.0))                 
    # Top edge of interface nodes    
    for xNodeIdx in range(1,numberOfSolidXElements*(numberOfNodesXi-1)+2):
        nodeNumber = xNodeIdx+numberOfSolidYElements*(numberOfNodesXi-1)
        #nodeDomain = interfaceDecomposition.NodeDomainGet(nodeNumber,1)
        nodeDomain = computationalNodeNumber
        if (nodeDomain == computationalNodeNumber):
            xPosition = fluidX1Size+float(xNodeIdx-1)/float(numberOfSolidXElements*(numberOfNodesXi-1))*solidXSize
            yPosition = solidYSize
            interfaceGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                             1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,xPosition)
            interfaceGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                             1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,yPosition)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
                print('         Position         = [ %.2f, %.2f ]' % (xPosition,yPosition))                 
            if (useHermite):
                interfaceGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,1,1.0)
                interfaceGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,2,0.0)
                if (debugLevel > 2):
                    print('        S1 derivative    = [ %.2f, %.2f ]' % (1.0,0.0))                 
    # Right edge of interface nodes    
    for yNodeIdx in range(1,numberOfSolidYElements*(numberOfNodesXi-1)+1):
        nodeNumber = yNodeIdx+(numberOfSolidYElements+numberOfSolidXElements)*(numberOfNodesXi-1)+1
        #nodeDomain = interfaceDecomposition.NodeDomainGet(nodeNumber,1)
        nodeDomain = computationalNodeNumber
        if (nodeDomain == computationalNodeNumber):
            xPosition = fluidX1Size+solidXSize
            yPosition = solidYSize-float(yNodeIdx)/float(numberOfSolidYElements*(numberOfNodesXi-1))*solidYSize
            interfaceGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                             1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,xPosition)
            interfaceGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                             1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,yPosition)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
                print('         Position         = [ %.2f, %.2f ]' % (xPosition,yPosition))                 
            if (useHermite):
                interfaceGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,1,0.0)
                interfaceGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                 1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,2,-1.0)
                if (debugLevel > 2):
                    print('        S1 derivative    = [ %.2f, %.2f ]' % (1.0,0.0))                 

    # Update fields            
    interfaceGeometricField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)
    interfaceGeometricField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)

if (progressDiagnostics):
    print('Geometric Parameters ... Done')

#================================================================================================================================
#  Equations Set
#================================================================================================================================

if (progressDiagnostics):
    print('Equations Sets ...')

if (problemType != FLUID):
    # Create the equations set for the solid region 
    solidEquationsSetField = iron.Field()
    solidEquationsSet = iron.EquationsSet()
    solidEquationsSetSpecification = [iron.EquationsSetClasses.ELASTICITY,
                                      iron.EquationsSetTypes.FINITE_ELASTICITY,
                                      iron.EquationsSetSubtypes.MOONEY_RIVLIN]
                                      #iron.EquationsSetSubtypes.MR_AND_GROWTH_LAW_IN_CELLML]
    solidEquationsSet.CreateStart(solidEquationsSetUserNumber,solidRegion,solidGeometricField,
                                  solidEquationsSetSpecification,solidEquationsSetFieldUserNumber,
                                  solidEquationsSetField)
    solidEquationsSet.OutputTypeSet(solidEquationsSetOutputType)
    solidEquationsSet.CreateFinish()
    
if (problemType != SOLID):
    # Create the equations set for the fluid region - ALE Navier-Stokes
    fluidEquationsSetField = iron.Field()
    fluidEquationsSet = iron.EquationsSet()
    if RBS:
        if (problemType == FSI):
            fluidEquationsSetSpecification = [iron.EquationsSetClasses.FLUID_MECHANICS,
                                              iron.EquationsSetTypes.NAVIER_STOKES_EQUATION,
                                              iron.EquationsSetSubtypes.ALE_RBS_NAVIER_STOKES]
        else:
            fluidEquationsSetSpecification = [iron.EquationsSetClasses.FLUID_MECHANICS,
                                              iron.EquationsSetTypes.NAVIER_STOKES_EQUATION,
                                              iron.EquationsSetSubtypes.TRANSIENT_RBS_NAVIER_STOKES]            
    else:
        if (problemType == FSI):
            fluidEquationsSetSpecification = [iron.EquationsSetClasses.FLUID_MECHANICS,
                                              iron.EquationsSetTypes.NAVIER_STOKES_EQUATION,
                                              iron.EquationsSetSubtypes.ALE_NAVIER_STOKES]
        else:
            fluidEquationsSetSpecification = [iron.EquationsSetClasses.FLUID_MECHANICS,
                                              iron.EquationsSetTypes.NAVIER_STOKES_EQUATION,
                                              iron.EquationsSetSubtypes.TRANSIENT_NAVIER_STOKES]
        
    fluidEquationsSet.CreateStart(fluidEquationsSetUserNumber,fluidRegion,fluidGeometricField,
                                  fluidEquationsSetSpecification,fluidEquationsSetFieldUserNumber,
                                  fluidEquationsSetField)
    fluidEquationsSet.OutputTypeSet(fluidEquationsSetOutputType)
    fluidEquationsSet.CreateFinish()

    if RBS:
        # Set boundary retrograde flow stabilisation scaling factor (default 0- do not use)
        fluidEquationsSetField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U1,
                                                           iron.FieldParameterSetTypes.VALUES,1,1.0)
        # Set max CFL number (default 1.0)
        fluidEquationsSetField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U1,
                                                           iron.FieldParameterSetTypes.VALUES,2,1.0E20)
        # Set time increment (default 0.0)
        fluidEquationsSetField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U1,
                                                           iron.FieldParameterSetTypes.VALUES,3,timeStep)
        # Set stabilisation type (default 1.0 = RBS)
        fluidEquationsSetField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U1,
                                                           iron.FieldParameterSetTypes.VALUES,4,1.0)
        
if (problemType == FSI):
    # Create the equations set for the moving mesh
    movingMeshEquationsSetField = iron.Field()
    movingMeshEquationsSet = iron.EquationsSet()
    movingMeshEquationsSetSpecification = [iron.EquationsSetClasses.CLASSICAL_FIELD,
                                           iron.EquationsSetTypes.LAPLACE_EQUATION,
                                           iron.EquationsSetSubtypes.MOVING_MESH_LAPLACE]
    movingMeshEquationsSet.CreateStart(movingMeshEquationsSetUserNumber,fluidRegion,fluidGeometricField,
                                       movingMeshEquationsSetSpecification,movingMeshEquationsSetFieldUserNumber,
                                       movingMeshEquationsSetField)
    movingMeshEquationsSet.OutputTypeSet(movingMeshEquationsSetOutputType)
    movingMeshEquationsSet.CreateFinish()
    
if (progressDiagnostics):
    print('Equations Sets ... Done')


#================================================================================================================================
#  Dependent Field
#================================================================================================================================

if (progressDiagnostics):
    print('Dependent Fields ...')

if (problemType != FLUID):
    # Create the equations set dependent field variables for the solid equations set
    solidDependentField = iron.Field()
    solidEquationsSet.DependentCreateStart(solidDependentFieldUserNumber,solidDependentField)
    solidDependentField.VariableLabelSet(iron.FieldVariableTypes.U,'SolidDependent')
    for componentIdx in range(1,numberOfDimensions+1):
        solidDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,componentIdx,1)
        solidDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.DELUDELN,componentIdx,1)
    solidDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,numberOfDimensions+1,2)
    solidDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.DELUDELN,numberOfDimensions+1,2)
    solidDependentField.ComponentInterpolationSet(iron.FieldVariableTypes.U,numberOfDimensions+1,iron.FieldInterpolationTypes.NODE_BASED)
    solidDependentField.ComponentInterpolationSet(iron.FieldVariableTypes.DELUDELN,numberOfDimensions+1,iron.FieldInterpolationTypes.NODE_BASED)
    if (useHermite):
        solidDependentField.ScalingTypeSet(iron.FieldScalingTypes.ARITHMETIC_MEAN)
    else:
        solidDependentField.ScalingTypeSet(iron.FieldScalingTypes.NONE)
    solidEquationsSet.DependentCreateFinish()

    # Initialise the solid dependent field from undeformed geometry and displacement bcs and set hydrostatic pressure
    for componentIdx in range(1,numberOfDimensions+1):
        solidGeometricField.ParametersToFieldParametersComponentCopy(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,\
                                                                     componentIdx,solidDependentField,iron.FieldVariableTypes.U,
                                                                     iron.FieldParameterSetTypes.VALUES,componentIdx)
    solidDependentField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                    numberOfDimensions+1,solidPInit)
    
    solidDependentField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)
    solidDependentField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)

if (problemType != FLUID):
    # Create the equations set dependent field variables for dynamic Navier-Stokes
    fluidDependentField = iron.Field()
    fluidEquationsSet.DependentCreateStart(fluidDependentFieldUserNumber,fluidDependentField)
    fluidDependentField.VariableLabelSet(iron.FieldVariableTypes.U,'FluidDependent')
    # Set the mesh component to be used by the field components.
    for componentIdx in range(1,numberOfDimensions+1):
        fluidDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,componentIdx,1)
        fluidDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.DELUDELN,componentIdx,1)
    fluidDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,numberOfDimensions+1,2)
    fluidDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.DELUDELN,numberOfDimensions+1,2)
    # fluidDependentField.ComponentInterpolationSet(iron.FieldVariableTypes.U,numberOfDimensions+1,iron.FieldInterpolationTypes.NODE_BASED)
    # fluidDependentField.ComponentInterpolationSet(iron.FieldVariableTypes.DELUDELN,numberOfDimensions+1,iron.FieldInterpolationTypes.NODE_BASED)
    # Finish the equations set dependent field variables
    fluidEquationsSet.DependentCreateFinish()

    # Initialise the fluid dependent field
    for componentIdx in range(1,numberOfDimensions+1):
        fluidDependentField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,componentIdx,0.0)
    # Initialise pressure component
    fluidDependentField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                    numberOfDimensions+1,fluidPInit)
    if RBS:
        fluidDependentField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.PRESSURE_VALUES,3,fluidPInit)
        
    fluidDependentField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)
    fluidDependentField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)

if (problemType == FSI):        
    # Create the equations set dependent field variables for moving mesh
    movingMeshDependentField = iron.Field()
    movingMeshEquationsSet.DependentCreateStart(movingMeshDependentFieldUserNumber,movingMeshDependentField)
    movingMeshDependentField.VariableLabelSet(iron.FieldVariableTypes.U,'MovingMeshDependent')
    # Set the mesh component to be used by the field components.
    for componentIdx in range(1,numberOfDimensions+1):
        movingMeshDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,componentIdx,1)
        movingMeshDependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.DELUDELN,componentIdx,1)
    # Finish the equations set dependent field variables
    movingMeshEquationsSet.DependentCreateFinish()

    # Initialise dependent field moving mesh
    for ComponentIdx in range(1,numberOfDimensions+1):
        movingMeshDependentField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES, \
                                                             componentIdx,0.0)

    movingMeshDependentField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)
    movingMeshDependentField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)

if (progressDiagnostics):
    print('Dependent Fields ... Done')
     
#================================================================================================================================
#  Materials Field
#================================================================================================================================

if (progressDiagnostics):
    print('Materials Fields ...')

if (problemType != FLUID):
    # Create the solid materials field
    solidMaterialsField = iron.Field()
    solidEquationsSet.MaterialsCreateStart(solidMaterialsFieldUserNumber,solidMaterialsField)
    solidMaterialsField.VariableLabelSet(iron.FieldVariableTypes.U,'SolidMaterials')
    solidMaterialsField.VariableLabelSet(iron.FieldVariableTypes.V,'SolidDensity')
    solidEquationsSet.MaterialsCreateFinish()
    # Set Mooney-Rivlin constants c10 and c01 respectively
    solidMaterialsField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,mooneyRivlin1)
    solidMaterialsField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,2,mooneyRivlin2)
    solidMaterialsField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.V,iron.FieldParameterSetTypes.VALUES,1,solidDensity)

if (problemType != SOLID):
    # Create the equations set materials field variables for dynamic Navier-Stokes
    fluidMaterialsField = iron.Field()
    fluidEquationsSet.MaterialsCreateStart(fluidMaterialsFieldUserNumber,fluidMaterialsField)
    # Finish the equations set materials field variables
    fluidEquationsSet.MaterialsCreateFinish()
    fluidMaterialsField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,fluidDynamicViscosity)
    fluidMaterialsField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,2,fluidDensity)
    
if (problemType == FSI):    
    # Create the equations set materials field variables for moving mesh
    movingMeshMaterialsField = iron.Field()
    movingMeshEquationsSet.MaterialsCreateStart(movingMeshMaterialsFieldUserNumber,movingMeshMaterialsField)
    # Finish the equations set materials field variables
    movingMeshEquationsSet.MaterialsCreateFinish()

    movingMeshMaterialsField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,\
                                                         movingMeshKParameter)
   
if (progressDiagnostics):
    print('Materials Fields ... Done')
    
#================================================================================================================================
#  Source Field
#================================================================================================================================

if (problemType != FLUID):
    if (gravityFlag):
        if (progressDiagnostics):
            print('Source Fields ...')
        #Create the source field with the gravity vector
        soidSourceField = iron.Field()
        solidEquationsSet.SourceCreateStart(solidSourceFieldUserNumber,solidSourceField)
        solidSourceField.ScalingTypeSet(iron.FieldScalingTypes.NONE)
        solidEquationsSet.SourceCreateFinish()
        for componentIdx in range(1,numberOfDimensions+1):
            solidSourceField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,component_Idx,
                                                         gravity[componentIdx-1])
        if (progressDiagnostics):
            print('Source Fields ... Done')
         
#================================================================================================================================
# Independent Field
#================================================================================================================================

if (problemType == FSI):
    if (progressDiagnostics):
        print('Independent Fields ...')

    # Create fluid mesh velocity independent field 
    fluidIndependentField = iron.Field()
    fluidEquationsSet.IndependentCreateStart(fluidIndependentFieldUserNumber,fluidIndependentField)
    fluidIndependentField.VariableLabelSet(iron.FieldVariableTypes.U,'FluidIndependent')
    # Set the mesh component to be used by the field components.
    for componentIdx in range(1,numberOfDimensions+1):
        fluidIndependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,componentIdx,1)
    # Finish the equations set independent field variables
    fluidEquationsSet.IndependentCreateFinish()
  
    # Create the moving mesh independent field 
    movingMeshIndependentField = iron.Field()
    movingMeshEquationsSet.IndependentCreateStart(movingMeshIndependentFieldUserNumber,movingMeshIndependentField)
    movingMeshIndependentField.VariableLabelSet(iron.FieldVariableTypes.U,'MovingMeshIndependent')
    # Set the mesh component to be used by the field components.
    for componentIdx in range(1,numberOfDimensions+1):
        movingMeshIndependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,componentIdx,1)    
    # Finish the equations set independent field variables
    movingMeshEquationsSet.IndependentCreateFinish()

    # Initialise independent field moving mesh
    movingMeshIndependentField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,movingMeshKParameter)

    if (progressDiagnostics):
        print('Independent Fields ... Done')

#================================================================================================================================
#  Equations
#================================================================================================================================

if (progressDiagnostics):
    print('Equations ...')

if (problemType != FLUID):
    # Solid equations
    solidEquations = iron.Equations()
    solidEquationsSet.EquationsCreateStart(solidEquations)
    solidEquations.sparsityType = iron.EquationsSparsityTypes.SPARSE
    solidEquations.outputType = solidEquationsOutputType
    solidEquationsSet.EquationsCreateFinish()

if (problemType != SOLID):
    # Fluid equations 
    fluidEquations = iron.Equations()
    fluidEquationsSet.EquationsCreateStart(fluidEquations)
    fluidEquations.sparsityType = iron.EquationsSparsityTypes.SPARSE
    fluidEquations.outputType = fluidEquationsOutputType
    fluidEquationsSet.EquationsCreateFinish()

if (problemType == FSI):
    # Moving mesh equations
    movingMeshEquations = iron.Equations()
    movingMeshEquationsSet.EquationsCreateStart(movingMeshEquations)
    movingMeshEquations.sparsityType = iron.EquationsSparsityTypes.SPARSE
    movingMeshEquations.outputType = movingMeshEquationsOutputType
    movingMeshEquationsSet.EquationsCreateFinish()

if (progressDiagnostics):
    print('Equations ... Done')

#================================================================================================================================
#  CellML
#================================================================================================================================

if (progressDiagnostics):
    print('CellML ...')

if (problemType != SOLID):
    # Create CellML equations for the temporal fluid boundary conditions
    bcCellML = iron.CellML()
    bcCellML.CreateStart(bcCellMLUserNumber,fluidRegion)
    bcCellMLIdx = bcCellML.ModelImport("exponentialrampupinletbc.cellml")
    bcCellML.VariableSetAsKnown(bcCellMLIdx,"main/A")
    bcCellML.VariableSetAsKnown(bcCellMLIdx,"main/B")
    bcCellML.VariableSetAsKnown(bcCellMLIdx,"main/C")
    bcCellML.VariableSetAsKnown(bcCellMLIdx,"main/x")
    bcCellML.VariableSetAsKnown(bcCellMLIdx,"main/y")
    bcCellML.VariableSetAsWanted(bcCellMLIdx,"main/inletx")
    bcCellML.VariableSetAsWanted(bcCellMLIdx,"main/inlety")
    bcCellML.CreateFinish()

    # Create CellML <--> OpenCMISS field maps
    bcCellML.FieldMapsCreateStart()
    # Map geometric field to x0 and y0
    bcCellML.CreateFieldToCellMLMap(fluidGeometricField,iron.FieldVariableTypes.U,1,iron.FieldParameterSetTypes.VALUES,
	                            bcCellMLIdx,"main/x",iron.FieldParameterSetTypes.VALUES)
    bcCellML.CreateFieldToCellMLMap(fluidGeometricField,iron.FieldVariableTypes.U,2,iron.FieldParameterSetTypes.VALUES,
	                            bcCellMLIdx,"main/y",iron.FieldParameterSetTypes.VALUES)
    # Map fluid velocity to ensure dependent field isn't cleared when the velocities are copied back
    bcCellML.CreateFieldToCellMLMap(fluidDependentField,iron.FieldVariableTypes.U,1,iron.FieldParameterSetTypes.VALUES,
	                            bcCellMLIdx,"main/inletx",iron.FieldParameterSetTypes.VALUES)
    bcCellML.CreateFieldToCellMLMap(fluidDependentField,iron.FieldVariableTypes.U,2,iron.FieldParameterSetTypes.VALUES,
	                            bcCellMLIdx,"main/inlety",iron.FieldParameterSetTypes.VALUES)
    # Map inletx and inlety to dependent field
    bcCellML.CreateCellMLToFieldMap(bcCellMLIdx,"main/inletx",iron.FieldParameterSetTypes.VALUES,
	                            fluidDependentField,iron.FieldVariableTypes.U,1,iron.FieldParameterSetTypes.VALUES)
    bcCellML.CreateCellMLToFieldMap(bcCellMLIdx,"main/inlety",iron.FieldParameterSetTypes.VALUES,
	                            fluidDependentField,iron.FieldVariableTypes.U,2,iron.FieldParameterSetTypes.VALUES)
    bcCellML.FieldMapsCreateFinish()


    # Create the CellML models field
    bcCellMLModelsField = iron.Field()
    bcCellML.ModelsFieldCreateStart(bcCellMLModelsFieldUserNumber,bcCellMLModelsField)
    bcCellMLModelsField.VariableLabelSet(iron.FieldVariableTypes.U,"BCModelMap")
    bcCellML.ModelsFieldCreateFinish()

    # Only evaluate BC on inlet nodes
    bcCellMLModelsField.ComponentValuesInitialiseIntg(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,0)
    if (debugLevel > 2):
        print('  CellML Boundary Conditions:')
        print('    Inlet Model Set:')
    for yNodeIdx in range(2,numberOfSolidYElements*(numberOfNodesXi-1)+1):
        nodeNumber = (yNodeIdx-1)*((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)+1
        nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,1)
        if (nodeDomain == computationalNodeNumber):
            bcCellMLModelsField.ParameterSetUpdateNodeIntg(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                           1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,1)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
    for yNodeIdx in range(1,numberOfFluidYElements*(numberOfNodesXi-1)+1):
        nodeNumber = numberOfSolidYElements*(numberOfNodesXi-1)*((numberOfFluidX1Elements+numberOfFluidX2Elements)* \
                                                                 (numberOfNodesXi-1)+2) + (yNodeIdx-1)* \
                                                                 ((numberOfFluidX1Elements+ numberOfSolidXElements+ \
                                                                   numberOfFluidX2Elements)*(numberOfNodesXi-1)+1)+1
        nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,1)
        if (nodeDomain == computationalNodeNumber):
            bcCellMLModelsField.ParameterSetUpdateNodeIntg(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                           1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,1)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))

    # Create the CellML state field
    bcCellMLStateField = iron.Field()
    bcCellML.StateFieldCreateStart(bcCellMLStateFieldUserNumber,bcCellMLStateField)
    bcCellMLStateField.VariableLabelSet(iron.FieldVariableTypes.U,"BCState")
    bcCellML.StateFieldCreateFinish()

    # Create the CellML parameters field
    bcCellMLParametersField = iron.Field()
    bcCellML.ParametersFieldCreateStart(bcCellMLParametersFieldUserNumber,bcCellMLParametersField)
    bcCellMLParametersField.VariableLabelSet(iron.FieldVariableTypes.U,"BCParameters")
    bcCellML.ParametersFieldCreateFinish()

    # Get the component numbers
    AComponentNumber = bcCellML.FieldComponentGet(bcCellMLIdx,iron.CellMLFieldTypes.PARAMETERS,"main/A")
    BComponentNumber = bcCellML.FieldComponentGet(bcCellMLIdx,iron.CellMLFieldTypes.PARAMETERS,"main/B")
    CComponentNumber = bcCellML.FieldComponentGet(bcCellMLIdx,iron.CellMLFieldTypes.PARAMETERS,"main/C")
    # Set up the parameters field
    bcCellMLParametersField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,AComponentNumber,A)
    bcCellMLParametersField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,BComponentNumber,B)
    bcCellMLParametersField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,CComponentNumber,C)

    # Create the CELL intermediate field
    bcCellMLIntermediateField = iron.Field()
    bcCellML.IntermediateFieldCreateStart(bcCellMLIntermediateFieldUserNumber,bcCellMLIntermediateField)
    bcCellMLIntermediateField.VariableLabelSet(iron.FieldVariableTypes.U,"BCIntermediate")
    bcCellML.IntermediateFieldCreateFinish()

if (progressDiagnostics):
    print('CellML ... Done')

#================================================================================================================================
#  Interface Condition
#================================================================================================================================

if (problemType == FSI):
    if (progressDiagnostics):
        print('Interface Conditions ...')

    # Create an interface condition between the two meshes
    interfaceCondition = iron.InterfaceCondition()
    interfaceCondition.CreateStart(interfaceConditionUserNumber,interface,interfaceGeometricField)
    # Specify the method for the interface condition
    interfaceCondition.MethodSet(iron.InterfaceConditionMethods.LAGRANGE_MULTIPLIERS)
    # Specify the type of interface condition operator
    interfaceCondition.OperatorSet(iron.InterfaceConditionOperators.SOLID_FLUID)
    # Add in the dependent variables from the equations sets
    interfaceCondition.DependentVariableAdd(solidMeshIndex,solidEquationsSet,iron.FieldVariableTypes.U)
    interfaceCondition.DependentVariableAdd(fluidMeshIndex,fluidEquationsSet,iron.FieldVariableTypes.U)
    # Set the label
    interfaceCondition.LabelSet("FSI Interface Condition")
    # Set the output type
    interfaceCondition.OutputTypeSet(interfaceConditionOutputType)
    # Finish creating the interface condition
    interfaceCondition.CreateFinish()

    if (progressDiagnostics):
        print('Interface Conditions ... Done')

    if (progressDiagnostics):
        print('Interface Lagrange Field ...')
    
    # Create the Lagrange multipliers field
    interfaceLagrangeField = iron.Field()
    interfaceCondition.LagrangeFieldCreateStart(interfaceLagrangeFieldUserNumber,interfaceLagrangeField)
    interfaceLagrangeField.VariableLabelSet(iron.FieldVariableTypes.U,'InterfaceLagrange')
    # Finish the Lagrange multipliers field
    interfaceCondition.LagrangeFieldCreateFinish()
    
    for componentIdx in range(1,numberOfDimensions+1):
        interfaceLagrangeField.ComponentValuesInitialise(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,componentIdx,0.0)

        interfaceLagrangeField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)
        interfaceLagrangeField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)

    if (progressDiagnostics):
        print('Interface Lagrange Field ... Done')

    if (progressDiagnostics):
        print('Interface Equations ...')

    # Create the interface condition equations
    interfaceEquations = iron.InterfaceEquations()
    interfaceCondition.EquationsCreateStart(interfaceEquations)
    # Set the interface equations sparsity
    interfaceEquations.sparsityType = iron.EquationsSparsityTypes.SPARSE
    # Set the interface equations output
    interfaceEquations.outputType = interfaceEquationsOutputType
    # Finish creating the interface equations
    interfaceCondition.EquationsCreateFinish()

    if (progressDiagnostics):
        print('Interface Equations ... Done')

#================================================================================================================================
#  Problem
#================================================================================================================================

if (progressDiagnostics):
    print('Problems ...')

# Create a FSI problem
fsiProblem = iron.Problem()
if (problemType == SOLID):
   fsiProblemSpecification = [iron.ProblemClasses.ELASTICITY,
                              iron.ProblemTypes.FINITE_ELASTICITY,
                              iron.ProblemSubtypes.QUASISTATIC_FINITE_ELASTICITY]
elif (problemType == FLUID):
    if RBS:
        fsiProblemSpecification = [iron.ProblemClasses.FLUID_MECHANICS,
                                   iron.ProblemTypes.NAVIER_STOKES_EQUATION,
                                   iron.ProblemSubtypes.TRANSIENT_RBS_NAVIER_STOKES]
    else:
        fsiProblemSpecification = [iron.ProblemClasses.FLUID_MECHANICS,
                                   iron.ProblemTypes.NAVIER_STOKES_EQUATION,
                                   iron.ProblemSubtypes.TRANSIENT_NAVIER_STOKES]
elif (problemType == FSI):
    if RBS:
        fsiProblemSpecification = [iron.ProblemClasses.MULTI_PHYSICS,
                                   iron.ProblemTypes.FINITE_ELASTICITY_NAVIER_STOKES,
                                   iron.ProblemSubtypes.FINITE_ELASTICITY_RBS_NAVIER_STOKES_ALE]
    else:
        fsiProblemSpecification = [iron.ProblemClasses.MULTI_PHYSICS,
                                   iron.ProblemTypes.FINITE_ELASTICITY_NAVIER_STOKES,
                                   iron.ProblemSubtypes.FINITE_ELASTICITY_NAVIER_STOKES_ALE]
        
fsiProblem.CreateStart(fsiProblemUserNumber,fsiProblemSpecification)
fsiProblem.CreateFinish()

if (progressDiagnostics):
    print('Problems ... Done')

#================================================================================================================================
#  Control Loop
#================================================================================================================================

if (progressDiagnostics):
    print('Control Loops ...')

# Create the fsi problem control loop
fsiControlLoop = iron.ControlLoop()
fsiProblem.ControlLoopCreateStart()
fsiProblem.ControlLoopGet([iron.ControlLoopIdentifiers.NODE],fsiControlLoop)
fsiControlLoop.LabelSet('TimeLoop')
fsiControlLoop.TimesSet(startTime,stopTime,timeStep)
fsiControlLoop.TimeOutputSet(outputFrequency)
fsiProblem.ControlLoopCreateFinish()

if (progressDiagnostics):
    print('Control Loops ... Done')

#================================================================================================================================
#  Solvers
#================================================================================================================================

if (progressDiagnostics):
    print('Solvers ...')

# Create the problem solver
bcCellMLEvaluationSolver = iron.Solver()
fsiDynamicSolver = iron.Solver()
fsiNonlinearSolver = iron.Solver()
fsiLinearSolver = iron.Solver()
movingMeshLinearSolver = iron.Solver()

fsiProblem.SolversCreateStart()
if (problemType == SOLID):
    # Solvers for growth Finite Elasticity problem
    # Get the BC CellML solver
    fsiProblem.SolverGet([iron.ControlLoopIdentifiers.NODE],1,bcCellMLEvaluationSolver)
    bcCellMLEvaluationSolver.outputType = iron.SolverOutputTypes.PROGRESS
    # Get the nonlinear solver
    fsiProblem.SolverGet([iron.ControlLoopIdentifiers.NODE],2,fsiNonlinearSolver)
    fsiNonlinearSolver.NewtonLineSearchTypeSet(iron.NewtonLineSearchTypes.LINEAR)
    #fsiNonlinearSolver.NewtonJacobianCalculationTypeSet(iron.JacobianCalculationTypes.EQUATIONS) #(.FD/EQUATIONS)
    fsiNonlinearSolver.NewtonJacobianCalculationTypeSet(iron.JacobianCalculationTypes.FD) #(.FD/EQUATIONS)
    fsiNonlinearSolver.NewtonMaximumFunctionEvaluationsSet(nonlinearMaxFunctionEvaluations)
    fsiNonlinearSolver.OutputTypeSet(fsiNonlinearSolverOutputType)
    fsiNonlinearSolver.NewtonAbsoluteToleranceSet(nonlinearAbsoluteTolerance)
    fsiNonlinearSolver.NewtonMaximumIterationsSet(nonlinearMaximumIterations)
    fsiNonlinearSolver.NewtonRelativeToleranceSet(nonlinearRelativeTolerance)
    fsiNonlinearSolver.NewtonLineSearchAlphaSet(nonlinearLinesearchAlpha)
    # Get the dynamic nonlinear linear solver
    fsiNonlinearSolver.NewtonLinearSolverGet(fsiLinearSolver)
    #fsiLinearSolver.LinearTypeSet(iron.LinearSolverTypes.ITERATIVE)
    #fsiLinearSolver.LinearIterativeTypeSet(iron.IterativeLinearSolverTypes.GMRES)
    #fsiLinearSolver.LinearIterativeGMRESRestartSet(linearRestartValue)
    #fsiLinearSolver.LinearIterativeMaximumIterationsSet(linearMaximumIterations)
    #fsiLinearSolver.LinearIterativeDivergenceToleranceSet(linearDivergenceTolerance)
    #fsiLinearSolver.LinearIterativeRelativeToleranceSet(linearRelativeTolerance)
    #fsiLinearSolver.LinearIterativeAbsoluteToleranceSet(linearAbsoluteTolerance)
    fsiLinearSolver.LinearTypeSet(iron.LinearSolverTypes.DIRECT)
    fsiLinearSolver.OutputTypeSet(fsiLinearSolverOutputType)
elif (problemType == FLUID):
    # Solvers for coupled FiniteElasticity NavierStokes problem
    # Get the BC CellML solver
    fsiProblem.SolverGet([iron.ControlLoopIdentifiers.NODE],1,bcCellMLEvaluationSolver)
    bcCellMLEvaluationSolver.outputType = iron.SolverOutputTypes.PROGRESS
    # Get the dynamic ALE solver
    fsiProblem.SolverGet([iron.ControlLoopIdentifiers.NODE],2,fsiDynamicSolver)
    fsiDynamicSolver.OutputTypeSet(fsiDynamicSolverOutputType)
    fsiDynamicSolver.DynamicThetaSet(fsiDynamicSolverTheta)
    # Get the dynamic nonlinear solver
    fsiDynamicSolver.DynamicNonlinearSolverGet(fsiNonlinearSolver)
    fsiNonlinearSolver.NewtonLineSearchTypeSet(iron.NewtonLineSearchTypes.LINEAR)
    fsiNonlinearSolver.NewtonJacobianCalculationTypeSet(iron.JacobianCalculationTypes.EQUATIONS) #(.FD/EQUATIONS)
    #fsiNonlinearSolver.NewtonJacobianCalculationTypeSet(iron.JacobianCalculationTypes.FD) #(.FD/EQUATIONS)
    fsiNonlinearSolver.NewtonMaximumFunctionEvaluationsSet(nonlinearMaxFunctionEvaluations)
    fsiNonlinearSolver.OutputTypeSet(fsiNonlinearSolverOutputType)
    fsiNonlinearSolver.NewtonAbsoluteToleranceSet(nonlinearAbsoluteTolerance)
    fsiNonlinearSolver.NewtonMaximumIterationsSet(nonlinearMaximumIterations)
    fsiNonlinearSolver.NewtonRelativeToleranceSet(nonlinearRelativeTolerance)
    fsiNonlinearSolver.NewtonLineSearchAlphaSet(nonlinearLinesearchAlpha)
    # Get the dynamic nonlinear linear solver
    fsiNonlinearSolver.NewtonLinearSolverGet(fsiLinearSolver)
    #fsiLinearSolver.LinearTypeSet(iron.LinearSolverTypes.ITERATIVE)
    #fsiLinearSolver.LinearIterativeMaximumIterationsSet(linearMaximumIterations)
    #fsiLinearSolver.LinearIterativeDivergenceToleranceSet(linearDivergenceTolerance)
    #fsiLinearSolver.LinearIterativeRelativeToleranceSet(linearRelativeTolerance)
    #fsiLinearSolver.LinearIterativeAbsoluteToleranceSet(linearAbsoluteTolerance)
    fsiLinearSolver.LinearTypeSet(iron.LinearSolverTypes.DIRECT)
    fsiLinearSolver.OutputTypeSet(fsiLinearSolverOutputType)
elif (problemType == FSI):
    # Solvers for coupled FiniteElasticity NavierStokes problem
    # Get the BC CellML solver
    fsiProblem.SolverGet([iron.ControlLoopIdentifiers.NODE],1,bcCellMLEvaluationSolver)
    bcCellMLEvaluationSolver.outputType = iron.SolverOutputTypes.PROGRESS
    # Get the dynamic ALE solver
    fsiProblem.SolverGet([iron.ControlLoopIdentifiers.NODE],2,fsiDynamicSolver)
    fsiDynamicSolver.OutputTypeSet(fsiDynamicSolverOutputType)
    fsiDynamicSolver.DynamicThetaSet(fsiDynamicSolverTheta)
    # Get the dynamic nonlinear solver
    fsiDynamicSolver.DynamicNonlinearSolverGet(fsiNonlinearSolver)
    fsiNonlinearSolver.NewtonLineSearchTypeSet(iron.NewtonLineSearchTypes.LINEAR)
    #fsiNonlinearSolver.NewtonJacobianCalculationTypeSet(iron.JacobianCalculationTypes.EQUATIONS) #(.FD/EQUATIONS)
    fsiNonlinearSolver.NewtonJacobianCalculationTypeSet(iron.JacobianCalculationTypes.FD) #(.FD/EQUATIONS)
    fsiNonlinearSolver.NewtonMaximumFunctionEvaluationsSet(nonlinearMaxFunctionEvaluations)
    fsiNonlinearSolver.OutputTypeSet(fsiNonlinearSolverOutputType)
    fsiNonlinearSolver.NewtonAbsoluteToleranceSet(nonlinearAbsoluteTolerance)
    fsiNonlinearSolver.NewtonMaximumIterationsSet(nonlinearMaximumIterations)
    fsiNonlinearSolver.NewtonRelativeToleranceSet(nonlinearRelativeTolerance)
    fsiNonlinearSolver.NewtonLineSearchAlphaSet(nonlinearLinesearchAlpha)
    # Get the dynamic nonlinear linear solver
    fsiNonlinearSolver.NewtonLinearSolverGet(fsiLinearSolver)
    #fsiLinearSolver.LinearTypeSet(iron.LinearSolverTypes.ITERATIVE)
    #fsiLinearSolver.LinearIterativeMaximumIterationsSet(linearMaximumIterations)
    #fsiLinearSolver.LinearIterativeDivergenceToleranceSet(linearDivergenceTolerance)
    #fsiLinearSolver.LinearIterativeRelativeToleranceSet(linearRelativeTolerance)
    #fsiLinearSolver.LinearIterativeAbsoluteToleranceSet(linearAbsoluteTolerance)
    fsiLinearSolver.LinearTypeSet(iron.LinearSolverTypes.DIRECT)
    fsiLinearSolver.OutputTypeSet(fsiLinearSolverOutputType)
    # Linear solver for moving mesh
    fsiProblem.SolverGet([iron.ControlLoopIdentifiers.NODE],3,movingMeshLinearSolver)
    movingMeshLinearSolver.OutputTypeSet(movingMeshLinearSolverOutputType)
# Finish the creation of the problem solver
fsiProblem.SolversCreateFinish()

if (progressDiagnostics):
    print('Solvers ... Done')

#================================================================================================================================
#  CellML Equations
#================================================================================================================================

if (progressDiagnostics):
    print('CellML Equations ...')

if (problemType != SOLID):
    # Create CellML equations and add BC equations to the solver
    bcEquations = iron.CellMLEquations()
    fsiProblem.CellMLEquationsCreateStart()
    bcCellMLEvaluationSolver.CellMLEquationsGet(bcEquations)
    bcEquationsIndex = bcEquations.CellMLAdd(bcCellML)
    fsiProblem.CellMLEquationsCreateFinish()

if (progressDiagnostics):
    print('CellML Equations ... Done')

#================================================================================================================================
#  Solver Equations
#================================================================================================================================

if (progressDiagnostics):
    print('Solver Equations ...')

# Start the creation of the fsi problem solver equations
fsiProblem.SolverEquationsCreateStart()
# Get the fsi dynamic solver equations
fsiSolverEquations = iron.SolverEquations()
if (problemType == SOLID):
    fsiNonlinearSolver.SolverEquationsGet(fsiSolverEquations)
else:
    fsiDynamicSolver.SolverEquationsGet(fsiSolverEquations)
fsiSolverEquations.sparsityType = iron.SolverEquationsSparsityTypes.SPARSE
if (problemType != FLUID):
    fsiSolidEquationsSetIndex = fsiSolverEquations.EquationsSetAdd(solidEquationsSet)
if (problemType != SOLID):
    fsiFluidEquationsSetIndex = fsiSolverEquations.EquationsSetAdd(fluidEquationsSet)
if (problemType == FSI):
    fsiInterfaceConditionIndex = fsiSolverEquations.InterfaceConditionAdd(interfaceCondition)
    # Set the time dependence of the interface matrix to determine the interface matrix coefficient in the solver matrix
    # (basiy position in big coupled matrix system)
    interfaceEquations.MatrixTimeDependenceTypeSet(fsiSolidEquationsSetIndex,True, \
                                                   [iron.InterfaceMatricesTimeDependenceTypes.STATIC,\
                                                    iron.InterfaceMatricesTimeDependenceTypes.FIRST_ORDER_DYNAMIC])
    interfaceEquations.MatrixTimeDependenceTypeSet(fsiFluidEquationsSetIndex,True, \
                                                   [iron.InterfaceMatricesTimeDependenceTypes.STATIC,\
                                                    iron.InterfaceMatricesTimeDependenceTypes.STATIC])
    
    # Create the moving mesh solver equations
    movingMeshSolverEquations = iron.SolverEquations()
    # Get the linear moving mesh solver equations
    movingMeshLinearSolver.SolverEquationsGet(movingMeshSolverEquations)
    movingMeshSolverEquations.sparsityType = iron.SolverEquationsSparsityTypes.SPARSE
    # Add in the equations set
    movingMeshEquationsSetIndex = movingMeshSolverEquations.EquationsSetAdd(movingMeshEquationsSet)
    
# Finish the creation of the fsi problem solver equations
fsiProblem.SolverEquationsCreateFinish()

if (progressDiagnostics):
    print('Solver Equations ...')

#================================================================================================================================
#  Boundary Conditions
#================================================================================================================================

if (progressDiagnostics):
    print('Boundary Conditions ...')

# Start the creation of the fsi boundary conditions
fsiBoundaryConditions = iron.BoundaryConditions()
fsiSolverEquations.BoundaryConditionsCreateStart(fsiBoundaryConditions)
if (problemType != FLUID):
    # Set no displacement boundary conditions on the bottom edge of the solid
    if (debugLevel > 2):
        print('  Solid Boundary Conditions:')
        print('    No Displacement Boundary conditions:')
    for xNodeIdx in range(1,numberOfSolidXElements*(numberOfNodesXi-1)+2):
        nodeNumber = xNodeIdx
        nodeDomain = solidDecomposition.NodeDomainGet(nodeNumber,1)
        if (nodeDomain == computationalNodeNumber):
            fsiBoundaryConditions.AddNode(solidDependentField,iron.FieldVariableTypes.U,1, \
                                          iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                          nodeNumber,1,iron.BoundaryConditionsTypes.FIXED,0.0)
            fsiBoundaryConditions.AddNode(solidDependentField,iron.FieldVariableTypes.U,1, \
                                          iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                          nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
                print('         Displacement      = [ %.2f, %.2f ]' % (0.0,0.0))                 
            if (useHermite):
                fsiBoundaryConditions.AddNode(solidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1, \
                                              nodeNumber,1,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.AddNode(solidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.AddNode(solidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2, \
                                              nodeNumber,1,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.AddNode(solidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.AddNode(solidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2, \
                                              nodeNumber,1,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.AddNode(solidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
    if (debugLevel > 2):
        print('    Reference Solid Pressure Boundary Condition:')
        nodeNumber = numberOfSolidXElements*(numberOfNodesXi-1)+1
        nodeDomain = solidDecomposition.NodeDomainGet(nodeNumber,1)
        if (nodeDomain == computationalNodeNumber):
            fsiBoundaryConditions.SetNode(solidDependentField,iron.FieldVariableTypes.U,1, \
                                          iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                          nodeNumber,3,iron.BoundaryConditionsTypes.FIXED,solidPRef)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
                print('         Pressure         =   %.2f' % (solidPRef))

if (problemType != SOLID):                
    # Set inlet boundary conditions on the left hand edge
    if (debugLevel > 2):
        print('  Fluid Boundary Conditions:')
        print('    Inlet Boundary conditions:')
    for yNodeIdx in range(2,numberOfSolidYElements*(numberOfNodesXi-1)+1):
        nodeNumber = (yNodeIdx-1)*((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)+1
        nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,1)
        if (nodeDomain == computationalNodeNumber):
            fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                          iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                          nodeNumber,1,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
            fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                          iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                          nodeNumber,2,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
                print('         Velocity         = [ %.2f, %.2f ]' % (0.0,0.0))                 
            if (useHermite):
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1, \
                                              nodeNumber,1,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2, \
                                              nodeNumber,1,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2, \
                                              nodeNumber,1,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
    for yNodeIdx in range(1,numberOfFluidYElements*(numberOfNodesXi-1)+1):
        nodeNumber = numberOfSolidYElements*(numberOfNodesXi-1)*((numberOfFluidX1Elements+numberOfFluidX2Elements)* \
                                                                 (numberOfNodesXi-1)+2) + (yNodeIdx-1)* \
                                                                 ((numberOfFluidX1Elements+ numberOfSolidXElements+ \
                                                                   numberOfFluidX2Elements)*(numberOfNodesXi-1)+1)+1
        nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,1)
        if (nodeDomain == computationalNodeNumber):
            fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                          iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                          nodeNumber,1,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
            fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                          iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                          nodeNumber,2,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
                print('         Velocity         = [ %.2f, %.2f ]' % (0.0,0.0))                 
            if (useHermite):
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1, \
                                              nodeNumber,1,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2, \
                                              nodeNumber,1,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2, \
                                              nodeNumber,1,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED_INLET,0.0)
    # Set outlet boundary conditions on the right hand edge to have zero pressure
    if (debugLevel > 2):
        print('    Outlet Boundary conditions:')
    # Elements to the right of the solid
    for yElementIdx in range(2,numberOfSolidYElements+1):
        nodeNumber = (yElementIdx-1)*(numberOfNodesXi-1)*((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)+\
                     (numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2
        nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,2)
        if (nodeDomain == computationalNodeNumber):
            if RBS:
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                              nodeNumber,numberOfDimensions+1,iron.BoundaryConditionsTypes.PRESSURE,fluidPRef)
            else:
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.DELUDELN,1, \
                                              iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                              nodeNumber,numberOfDimensions+1,iron.BoundaryConditionsTypes.FIXED,fluidPRef)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
                print('         Pressure         =   %.2f' % (fluidPRef))                 
        if RBS:
            # Set the element normals for outlet stabilisation
            elementNumber = numberOfFluidX1Elements+numberOfFluidX2Elements+(yElementIdx-2)*(numberOfFluidX1Elements+numberOfFluidX2Elements)
            elementDomain = fluidDecomposition.ElementDomainGet(elementNumber)
            if (elementDomain == computationalNodeNumber):
                # Set the outflow normal to (0,0,+1)
                fluidEquationsSetField.ParameterSetUpdateElementDP(iron.FieldVariableTypes.V, \
                                                                   iron.FieldParameterSetTypes.VALUES, \
                                                                   elementNumber,5,+1.0)
                fluidEquationsSetField.ParameterSetUpdateElementDP(iron.FieldVariableTypes.V, \
                                                                   iron.FieldParameterSetTypes.VALUES, \
                                                                   elementNumber,6,0.0)
                # Set the boundary type
                fluidEquationsSetField.ParameterSetUpdateElementDP(iron.FieldVariableTypes.V, \
                                                                   iron.FieldParameterSetTypes.VALUES, \
                                                                   elementNumber,9, \
                                                                   iron.BoundaryConditionsTypes.PRESSURE)                                                
                if (debugLevel > 2):
                    print('      Element     %d:' % (elementNumber))
                    print('         Normal          = [ %.2f, %.2f ]' % (+1.0,0.0))
    # Elements above the solid
    for yElementIdx in range(1,numberOfFluidYElements+2):
        nodeNumber = numberOfSolidYElements*(numberOfNodesXi-1)*((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2) \
                     + ((yElementIdx-1)*(numberOfNodesXi-1)+1)*((numberOfFluidX1Elements+numberOfSolidXElements+numberOfFluidX2Elements)*\
                                                            (numberOfNodesXi-1)+1)
        nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,2)
        if (nodeDomain == computationalNodeNumber):
            if RBS:
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                              nodeNumber,numberOfDimensions+1,iron.BoundaryConditionsTypes.PRESSURE,fluidPRef)
            else:
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.DELUDELN,1, \
                                              iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                              nodeNumber,numberOfDimensions+1,iron.BoundaryConditionsTypes.FIXED,fluidPRef)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
                print('         Pressure         =   %.2f' % (fluidPRef))
        if RBS:
            # Set the element normals for outlet stabilisation
            elementNumber = (numberOfFluidX1Elements+numberOfFluidX2Elements)*numberOfSolidYElements+\
                            numberOfFluidX1Elements+numberOfFluidX2Elements+numberOfSolidXElements+\
                            (yElementIdx-2)*(numberOfFluidX1Elements+numberOfFluidX2Elements+numberOfSolidXElements)
            elementDomain = fluidDecomposition.ElementDomainGet(elementNumber)
            if (elementDomain == computationalNodeNumber):
                # Set the outflow normal to (0,0,+1)
                fluidEquationsSetField.ParameterSetUpdateElementDP(iron.FieldVariableTypes.V, \
                                                                   iron.FieldParameterSetTypes.VALUES, \
                                                                   elementNumber,5,+1.0)
                fluidEquationsSetField.ParameterSetUpdateElementDP(iron.FieldVariableTypes.V, \
                                                                   iron.FieldParameterSetTypes.VALUES, \
                                                                   elementNumber,6,0.0)
                # Set the boundary type
                fluidEquationsSetField.ParameterSetUpdateElementDP(iron.FieldVariableTypes.V, \
                                                                   iron.FieldParameterSetTypes.VALUES, \
                                                                   elementNumber,9, \
                                                                   iron.BoundaryConditionsTypes.PRESSURE)
                if (debugLevel > 2):
                        print('      Element     %d:' % (elementNumber))
                        print('         Normal          = [ %.2f, %.2f ]' % (+1.0,0.0))
                
    # Set no-slip boundary conditions on the bottom edge
    if (debugLevel > 2):
        print('    No-slip Boundary conditions:')
    for xNodeIdx in range(1,numberOfFluidX1Elements*(numberOfNodesXi-1)+2):
        nodeNumber = xNodeIdx
        nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,1)
        if (nodeDomain == computationalNodeNumber):
            fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                          iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                          nodeNumber,1,iron.BoundaryConditionsTypes.FIXED,0.0)
            fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                          iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                          nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
            if (useHermite):
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1, \
                                              nodeNumber,1,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2, \
                                              nodeNumber,1,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2, \
                                              nodeNumber,1,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
    for xNodeIdx in range(1,numberOfFluidX2Elements*(numberOfNodesXi-1)+2):
        nodeNumber = numberOfFluidX1Elements*(numberOfNodesXi-1)+1+xNodeIdx
        nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,1)
        if (nodeDomain == computationalNodeNumber):
            fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                          iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                          nodeNumber,1,iron.BoundaryConditionsTypes.FIXED,0.0)
            fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                          iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                          nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
            if (useHermite):
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1, \
                                              nodeNumber,1,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2, \
                                              nodeNumber,1,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2, \
                                              nodeNumber,1,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
    if (problemType == FLUID):
        # Set no slip around the solid
        # Left and right solid edge nodes
        for yNodeIdx in range(2,numberOfSolidYElements*(numberOfNodesXi-1)+1):
            nodeNumber1 = (yNodeIdx-1)*((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)+ \
                     numberOfFluidX1Elements*(numberOfNodesXi-1)+1
            nodeNumber2 = nodeNumber1+1
            nodeDomain1 = fluidDecomposition.NodeDomainGet(nodeNumber1,1)
            nodeDomain2 = fluidDecomposition.NodeDomainGet(nodeNumber2,1)
            if (nodeDomain1 == computationalNodeNumber):
                fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                              iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber1,1,
                                              iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                              iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber1,2,
                                              iron.BoundaryConditionsTypes.FIXED,0.0)
                if (debugLevel > 2):
                    print('      Node        %d:' % (nodeNumber1))
                if (useHermite):    
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber1,1,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber1,1,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber1,1,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber1,2,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber1,2,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber1,2,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
            if (nodeDomain2 == computationalNodeNumber):
                fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                              iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber2,1,
                                              iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                              iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber2,2,
                                              iron.BoundaryConditionsTypes.FIXED,0.0)
                if (debugLevel > 2):
                    print('      Node        %d:' % (nodeNumber2))
                if (useHermite):    
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber2,1,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber2,1,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber2,1,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber2,2,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber2,2,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber2,2,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
        # Top solid edge nodes
        for xNodeIdx in range(1,numberOfSolidXElements*(numberOfNodesXi-1)+2):
            nodeNumber = xNodeIdx+numberOfSolidYElements*(numberOfNodesXi-1)*((numberOfFluidX1Elements+numberOfFluidX2Elements)* \
                      (numberOfNodesXi-1)+2)+numberOfFluidX1Elements*(numberOfNodesXi-1)
            nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,1)
            if (nodeDomain == computationalNodeNumber):
                fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                              iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,
                                              iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                              iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,
                                              iron.BoundaryConditionsTypes.FIXED,0.0)
                if (debugLevel > 2):
                    print('      Node        %d:' % (nodeNumber))
                if (useHermite):    
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,1,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,1,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,1,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,2,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,2,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
                    fsiBoundaryConditions.AddNode(fluidDependentField,iron.FieldVariableTypes.U,1,
                                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,2,
                                                  iron.BoundaryConditionsTypes.FIXED,0.0)
    # Set slip boundary conditions on the top edge
    if (debugLevel > 2):
        print('    Slip Boundary conditions:')
    for xNodeIdx in range(1,(numberOfFluidX1Elements+numberOfSolidXElements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2):
        nodeNumber = ((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)*numberOfSolidYElements*(numberOfNodesXi-1)+ \
                     ((numberOfFluidX1Elements+numberOfSolidXElements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+1)* \
                     numberOfFluidYElements*(numberOfNodesXi-1)+xNodeIdx
        nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,1)
        if (nodeDomain == computationalNodeNumber):
            fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                          iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                          nodeNumber,1,iron.BoundaryConditionsTypes.FIXED,0.0)
            fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                          iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                          nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
            if (useHermite):
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
                fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                              iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2, \
                                              nodeNumber,2,iron.BoundaryConditionsTypes.FIXED,0.0)
    if (debugLevel > 2):
        print('    Reference Fluid Pressure Boundary Condition:')
        nodeNumber = (numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2
        nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,2)
        if (nodeDomain == computationalNodeNumber):
            fsiBoundaryConditions.SetNode(fluidDependentField,iron.FieldVariableTypes.U,1, \
                                          iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV, \
                                          nodeNumber,3,iron.BoundaryConditionsTypes.FIXED,fluidPRef)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
                print('         Pressure         =   %.2f' % (fluidPRef))

if (problemType == FSI):
    # Remove dof's at nodes where solid displacement and zero velocity is set (first n last interface node)
    if (debugLevel > 2):
        print('  Lagrange Boundary Conditions:')
        print('    Fixed Boundary conditions:')
    fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                  iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,1,1, \
                                  iron.BoundaryConditionsTypes.FIXED,0.0)
    fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                  iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,1,2, \
                                  iron.BoundaryConditionsTypes.FIXED,0.0)
    fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                  iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,numberOfInterfaceNodes,1, \
                                  iron.BoundaryConditionsTypes.FIXED,0.0)
    fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                  iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,numberOfInterfaceNodes,2, \
                                  iron.BoundaryConditionsTypes.FIXED,0.0)
    if (debugLevel > 2):
        print('      Node        %d:' % (1))
        print('      Node        %d:' % (numberOfInterfaceNodes))
    if (useHermite):
        fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                      iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,1,1, \
                                      iron.BoundaryConditionsTypes.FIXED,0.0)
        fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                      iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,1,1, \
                                      iron.BoundaryConditionsTypes.FIXED,0.0)
        fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                      iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,1,1, \
                                      iron.BoundaryConditionsTypes.FIXED,0.0)
        fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                      iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,1,2, \
                                      iron.BoundaryConditionsTypes.FIXED,0.0)
        fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                      iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,1,2, \
                                      iron.BoundaryConditionsTypes.FIXED,0.0)
        fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                      iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,1,2, \
                                      iron.BoundaryConditionsTypes.FIXED,0.0)
        fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                      iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,numberOfInterfaceNodes,1, \
                                      iron.BoundaryConditionsTypes.FIXED,0.0)
        fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                      iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,numberOfInterfaceNodes,1, \
                                      iron.BoundaryConditionsTypes.FIXED,0.0)
        fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                      iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,numberOfInterfaceNodes,1, \
                                      iron.BoundaryConditionsTypes.FIXED,0.0)
        fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                      iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,numberOfInterfaceNodes,2, \
                                      iron.BoundaryConditionsTypes.FIXED,0.0)
        fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                      iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,numberOfInterfaceNodes,2, \
                                      iron.BoundaryConditionsTypes.FIXED,0.0)
        fsiBoundaryConditions.SetNode(interfaceLagrangeField,iron.FieldVariableTypes.U,1, \
                                      iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,numberOfInterfaceNodes,2, \
                                      iron.BoundaryConditionsTypes.FIXED,0.0)               
# Finish FSI boundary conditions
fsiSolverEquations.BoundaryConditionsCreateFinish()

if (problemType == FSI):
    # Start the creation of the moving mesh boundary conditions
    movingMeshBoundaryConditions = iron.BoundaryConditions()
    movingMeshSolverEquations.BoundaryConditionsCreateStart(movingMeshBoundaryConditions)
    if (debugLevel > 2):
        print('  Moving Mesh Boundary Conditions:')
        print('    Fixed Wall Boundary conditions:')
    # Bottom edge nodes
    for xNodeIdx in range(1,(numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+3):
        nodeNumber = xNodeIdx
        nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,1)
        if (nodeDomain == computationalNodeNumber):
            movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,
                                                iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
            movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,
                                                iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
            if (useHermite):    
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
    # Side edges nodes
    for yNodeIdx in range(2,numberOfSolidYElements*(numberOfNodesXi-1)+1):
        nodeNumber1 = (yNodeIdx-1)*((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)+1
        nodeNumber2 = yNodeIdx*((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)
        nodeDomain1 = fluidDecomposition.NodeDomainGet(nodeNumber1,1)
        nodeDomain2 = fluidDecomposition.NodeDomainGet(nodeNumber2,1)
        if (nodeDomain1 == computationalNodeNumber):
            movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber1,1,
                                                iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
            movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber1,2,
                                                iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber1))
            if (useHermite):    
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber1,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber1,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber1,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber1,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber1,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber1,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
        if (nodeDomain2 == computationalNodeNumber):
            movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber2,1,
                                                iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
            movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber2,2,
                                                iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber2))
            if (useHermite):    
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber2,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber2,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber2,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber2,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber2,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber2,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
    for yNodeIdx in range(1,numberOfFluidYElements*(numberOfNodesXi-1)+1):
        nodeNumber1 = (yNodeIdx-1)*((numberOfFluidX1Elements+numberOfSolidXElements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+1)+1+\
                      ((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)*\
                      numberOfSolidYElements*(numberOfNodesXi-1)
        nodeNumber2 = yNodeIdx*((numberOfFluidX1Elements+numberOfSolidXElements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+1)+\
                      ((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)*\
                      numberOfSolidYElements*(numberOfNodesXi-1)
        nodeDomain1 = fluidDecomposition.NodeDomainGet(nodeNumber1,1)
        nodeDomain2 = fluidDecomposition.NodeDomainGet(nodeNumber2,1)
        if (nodeDomain1 == computationalNodeNumber):
            movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber1,1,
                                                iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
            movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber1,2,
                                                iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber1))
            if (useHermite):    
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber1,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber1,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber1,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber1,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber1,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber1,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
        if (nodeDomain2 == computationalNodeNumber):
            movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber2,1,
                                                iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
            movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber2,2,
                                                iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber2))
            if (useHermite):    
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber2,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber2,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber2,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber2,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber2,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber2,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
    # Top edge nodes
    for xNodeIdx in range(1,(numberOfFluidX1Elements+numberOfSolidXElements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2):
        nodeNumber = xNodeIdx +((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)* \
                     numberOfSolidYElements*(numberOfNodesXi-1)+numberOfFluidYElements*(numberOfNodesXi-1)* \
                     ((numberOfFluidX1Elements+ numberOfSolidXElements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+1)
        nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,1)
        if (nodeDomain == computationalNodeNumber):
            movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,
                                                iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
            movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,
                                                iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
            if (useHermite):    
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,1,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
                movingMeshBoundaryConditions.SetNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,2,
                                                    iron.BoundaryConditionsTypes.FIXED_WALL,0.0)
    if (debugLevel > 2):
        print('    Moving Wall Boundary conditions:')
    # Left and right solid edge nodes
    for yNodeIdx in range(2,numberOfSolidYElements*(numberOfNodesXi-1)+1):
        nodeNumber1 = (yNodeIdx-1)*((numberOfFluidX1Elements+numberOfFluidX2Elements)*(numberOfNodesXi-1)+2)+ \
                     numberOfFluidX1Elements*(numberOfNodesXi-1)+1
        nodeNumber2 = nodeNumber1+1
        nodeDomain1 = fluidDecomposition.NodeDomainGet(nodeNumber1,1)
        nodeDomain2 = fluidDecomposition.NodeDomainGet(nodeNumber2,1)
        if (nodeDomain1 == computationalNodeNumber):
            movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber1,1,
                                                iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
            movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber1,2,
                                                iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber1))
            if (useHermite):    
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber1,1,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber1,1,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber1,1,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber1,2,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber1,2,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber1,2,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
        if (nodeDomain2 == computationalNodeNumber):
            movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber2,1,
                                                iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
            movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber2,2,
                                                iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber2))
            if (useHermite):    
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber2,1,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber2,1,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber2,1,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber2,2,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber2,2,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber2,2,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
    # Top solid edge nodes
    for xNodeIdx in range(1,numberOfSolidXElements*(numberOfNodesXi-1)+2):
        nodeNumber = xNodeIdx+numberOfSolidYElements*(numberOfNodesXi-1)*((numberOfFluidX1Elements+numberOfFluidX2Elements)* \
                      (numberOfNodesXi-1)+2)+numberOfFluidX1Elements*(numberOfNodesXi-1)
        nodeDomain = fluidDecomposition.NodeDomainGet(nodeNumber,1)
        if (nodeDomain == computationalNodeNumber):
            movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,
                                                iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
            movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,
                                                iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
            if (debugLevel > 2):
                print('      Node        %d:' % (nodeNumber))
            if (useHermite):    
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,1,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,1,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,1,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,2,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,2,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)
                movingMeshBoundaryConditions.AddNode(movingMeshDependentField,iron.FieldVariableTypes.U,1,
                                                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeNumber,2,
                                                    iron.BoundaryConditionsTypes.MOVED_WALL,0.0)

    # Finish moving mesh boundary conditions
    movingMeshSolverEquations.BoundaryConditionsCreateFinish()

if (progressDiagnostics):
    print('Boundary Conditions ... Done')

#================================================================================================================================
#  Run Solvers
#================================================================================================================================

#quit()

# Solve the problem
print('Solving problem...')
start = time.time()
fsiProblem.Solve()
end = time.time()
elapsed = end - start
print('Calculation Time = %3.4f' %elapsed)
print('Problem solved!')
print('#')

#================================================================================================================================
#  Finish Program
#================================================================================================================================

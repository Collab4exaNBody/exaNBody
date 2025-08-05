import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

def write_ovito_grid(filename, data_array, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), ordering='F'):
    """
    Writes a 3D NumPy array to a VTK structured points file.

    Parameters:
        filename (str): The output VTK file name (e.g., 'points.vtk').
        data_array (np.ndarray): A 3D NumPy array of shape (Nx, Ny, Nz).
        origin (tuple): The origin of the grid (default is (0.0, 0.0, 0.0)).
        spacing (tuple): The spacing between grid points (default is (1.0, 1.0, 1.0)).
    """
    if len(data_array.shape) != 3:
        raise ValueError("Input array must be 3D.")

    Nx, Ny, Nz = data_array.shape

    ofile = open(filename,'w')
    ofile.write('ITEM: TIMESTEP\n')
    ofile.write('0\n')
    ofile.write('ITEM: BOX BOUNDS pp pp pp\n')
    ofile.write('0 %d\n' %(Nx))
    ofile.write('0 %d\n' %(Ny))
    ofile.write('0 %d\n' %(Nz))
    ofile.write('ITEM: DIMENSION\n')
    ofile.write('3\n')
    ofile.write('ITEM: GRID SIZE nx ny nz\n')
    ofile.write('%d %d %d\n' %(Nx,Ny,Nz))
    ofile.write('ITEM: GRID CELLS MASK\n')
    data_array = data_array.ravel(order=ordering)
    for data in data_array:
        ofile.write('%5.8f\n' %(data))
    ofile.close()

def write_vtk_structured_points(filename, data_array, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), ordering='F'):
    """
    Writes a 3D NumPy array to a VTK structured points file.

    Parameters:
        filename (str): The output VTK file name (e.g., 'points.vtk').
        data_array (np.ndarray): A 3D NumPy array of shape (Nx, Ny, Nz).
        origin (tuple): The origin of the grid (default is (0.0, 0.0, 0.0)).
        spacing (tuple): The spacing between grid points (default is (1.0, 1.0, 1.0)).
    """
    if len(data_array.shape) != 3:
        raise ValueError("Input array must be 3D.")

    Nx, Ny, Nz = data_array.shape

    # Create the structured points
    structured_points = vtk.vtkImageData()
    structured_points.SetDimensions(Nx, Ny, Nz)
    structured_points.SetOrigin(origin)
    structured_points.SetSpacing(spacing)

    # Convert the NumPy array to a VTK array
    vtk_data_array = numpy_to_vtk(data_array.ravel(order=ordering), deep=True)
    vtk_data_array.SetName("ScalarData")

    # Add the data to the structured points
    structured_points.GetPointData().SetScalars(vtk_data_array)

    # Write the structured points to a file
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(filename)
    writer.SetInputData(structured_points)
    writer.Write()

import numpy as np

def populate_3d_array_with_spheres(Nx, Ny, Nz, num_spheres, mean_radius, radius_variance, seed=None):
    """
    Create a 3D numpy array with dimensions Nx, Ny, Nz, populate it with 0s, and
    fill it with 1s in N spheres of random radii following a Gaussian distribution,
    ensuring periodicity.

    Parameters:
        Nx, Ny, Nz (int): Dimensions of the 3D array.
        num_spheres (int): Number of spheres to populate.
        mean_radius (float): Mean radius of the spheres.
        radius_variance (float): Variance of the sphere radii.
        seed (int, optional): Seed for random number generator.

    Returns:
        numpy.ndarray: The populated 3D array.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize the array with zeros
    array = np.zeros((Nx, Ny, Nz), dtype=int)

    # Generate random sphere properties
    for _ in range(num_spheres):
        
        # Generate center in reduced coordinates [0, 1)
        cx_red = np.random.rand()
        cy_red = np.random.rand()
        cz_red = np.random.rand()
        
        # Scale to actual grid coordinates
        cx = int(np.floor(cx_red * Nx))
        cy = int(np.floor(cy_red * Ny))
        cz = int(np.floor(cz_red * Nz))
        
        print(cx,cy,cz)
        
        # Random radius from Gaussian distribution
        radius = max(1, np.random.normal(mean_radius*min(Nx,Ny,Nz), np.sqrt(radius_variance*Nx)))

        # Fill the sphere into the array
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    # Compute periodic distance to the sphere's center
                    dx = min(abs(x - cx), Nx - abs(x - cx))
                    dy = min(abs(y - cy), Ny - abs(y - cy))
                    dz = min(abs(z - cz), Nz - abs(z - cz))
                    distance = np.sqrt(dx**2 + dy**2 + dz**2)

                    if distance <= radius:
                        array[x, y, z] = 1

    return array
    
# Example usage
if __name__ == "__main__":
    # Create a sample 3D NumPy array
    Nx, Ny, Nz = 120,100,70
    num_spheres = 10
    mean_radius = 0.25
    radius_variance = 0.1
    data = populate_3d_array_with_spheres(Nx, Ny, Nz, num_spheres, mean_radius, radius_variance, seed=165)

    # Write the data to a VTK file
    ordering='F'
    write_vtk_structured_points("points_%dx%dx%d_%sorder.vtk" %(Nx,Ny,Nz,ordering), data, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0),ordering=ordering)
    write_ovito_grid("points_%dx%dx%d_%sorder.grid" %(Nx,Ny,Nz,ordering), data, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), ordering=ordering)

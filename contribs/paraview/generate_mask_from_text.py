import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pyevtk.hl import gridToVTK
import vtk
from vtk.util.numpy_support import numpy_to_vtk

def generate_exaNBody_mask(width, height, margin, font_path="DejaVuSans-Bold.ttf", max_font_size=None):
    text = "exaNBody"
    
    # Available area for the text
    available_width = width - 2 * margin
    available_height = height - 2 * margin

    # Try different font sizes to find the largest one that fits
    if max_font_size is None:
        max_font_size = min(available_width, available_height)
        
    for font_size in range(max_font_size, 1, -1):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            raise ValueError(f"Font not found: {font_path}")
        
        dummy_img = Image.new("L", (width, height))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w <= available_width and text_h <= available_height:
            break
    else:
        raise ValueError("Cannot fit text within given dimensions and margin.")

    # Create image and draw text
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    x = margin + (available_width - text_w) // 2
    y = margin + (available_height - text_h) // 2
    draw.text((x, y), text, fill=255, font=font)

    # Convert image to binary mask
    mask = np.array(img)
    binary_mask = (mask > 128).astype(int)
    return binary_mask

# Example usage
width, height = 200, 100
margin = 5
mask = generate_exaNBody_mask(width, height, margin)

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

def stack_2d_to_3d(array_2d, num_layers):
    """
    Create a 3D NumPy array by stacking a 2D array along the third dimension.

    Parameters:
        array_2d (ndarray): Input 2D NumPy array of shape (H, W)
        num_layers (int): Number of layers to replicate along the third (z) axis

    Returns:
        ndarray: 3D NumPy array of shape (num_layers, H, W)
    """
    databis = np.transpose(array_2d)
    datater = np.repeat(databis[:, :, np.newaxis], num_layers, axis=2)[:, ::-1, :]
    return datater

D3mask = stack_2d_to_3d(mask, 20)
ordering='F'
write_vtk_structured_points("test_mask_%sorder.vtk" %(ordering), D3mask, ordering=ordering)

# Display
plt.imshow(mask, cmap="gray")
plt.axis("off")
plt.title('"exaNBody" mask')
plt.savefig('image_mask.png')
plt.show()

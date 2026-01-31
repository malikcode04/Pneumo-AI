import pydicom
import numpy as np
from PIL import Image
import io

def read_dicom(file) -> dict:
    """
    Reads a DICOM file and returns the image (PIL) and metadata.
    
    Args:
        file: The uploaded file object.
        
    Returns:
        dict: {
            'image': PIL.Image,
            'metadata': dict (PatientID, StudyDate, etc.),
            'error': str (optional)
        }
    """
    try:
        # Read DICOM
        ds = pydicom.dcmread(file)
        
        # Extract pixel array
        pixel_array = ds.pixel_array.astype(float)
        
        # Rescale Intercept/Slope if present
        if hasattr(ds, 'RescaleIntercept') and hasattr(ds, 'RescaleSlope'):
            intercept = ds.RescaleIntercept
            slope = ds.RescaleSlope
            pixel_array = pixel_array * slope + intercept
            
        # Windowing (if present, otherwise min/max)
        # Simple normalization to 0-255 for display/inference
        # A more robust medical viewer uses WindowCenter/WindowWidth tags, 
        # but for AI inference which expects normalized inputs, min/max is often safe for a generic loader
        # unless specifically trained on Hounsfield units. 
        # Since the model expects standard images, we normalize to 0-255.
        
        pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
        pixel_array = (pixel_array * 255).astype(np.uint8)
        
        # Handle Photometric Interpretation (Monochrome1 vs Monochrome2)
        # MONOCHROME1: 0 is white, 1 is black
        # MONOCHROME2: 0 is black, 1 is white (standard)
        if hasattr(ds, 'PhotometricInterpretation'):
             if ds.PhotometricInterpretation == "MONOCHROME1":
                pixel_array = 255 - pixel_array
                
        # Convert to PIL (Grayscale)
        image = Image.fromarray(pixel_array).convert('RGB') # Convert to RGB as pipeline expects 3 channels
        
        # Meta data
        metadata = {
            "Patient ID": getattr(ds, "PatientID", "Unknown"),
            "Patient Name": str(getattr(ds, "PatientName", "Unknown")),
            "Study Date": getattr(ds, "StudyDate", "Unknown"),
            "Modality": getattr(ds, "Modality", "Unknown"),
            "Body Part": getattr(ds, "BodyPartExamined", "Unknown")
        }
        
        return {
            "image": image,
            "metadata": metadata,
            "error": None
        }

    except Exception as e:
        return {
            "image": None,
            "metadata": {},
            "error": str(e)
        }

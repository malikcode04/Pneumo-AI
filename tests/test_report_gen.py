import sys
import os
from pathlib import Path
from PIL import Image
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.report_gen import create_pdf_report

def test_pdf_generation():
    print("Starting PDF generation test...")
    
    # Mock data
    metadata = {
        "Name": "John Doe",
        "Patient ID": "TEST-123",
        "Age": "45",
        "Gender": "M",
        "Study Date": "2026-02-01",
        "Modality": "XR"
    }
    
    prediction_results = {
        "predicted_label": "Bacterial Pneumonia",
        "confidence": 0.85,
        "clinical_metrics": {
            "sensitivity": "92%",
            "specificity": "89%"
        },
        "uncertainty": {
            "entropy": 0.123
        }
    }
    
    # Create dummy images
    img1 = Image.new('RGB', (512, 512), color=(255, 255, 255))
    img2 = Image.new('RGB', (512, 512), color=(200, 200, 200))
    images = {
        'original': img1,
        'overlay': img2
    }
    
    doctor_name = "Smith"
    doctor_statement = "Patient shows significant opacity in the lower left lobe consistent with bacterial pneumonia."
    view_position = "PA"
    
    try:
        pdf_bytes = create_pdf_report(
            metadata, 
            prediction_results, 
            images, 
            doctor_name=doctor_name, 
            doctor_statement=doctor_statement,
            view_position=view_position
        )
        
        output_path = Path("tests/test_report.pdf")
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
            
        print(f"Success! PDF generated at: {output_path.absolute()}")
        return True
    except Exception as e:
        print(f"Failed! Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if not os.path.exists("tests"):
        os.makedirs("tests")
    test_pdf_generation()

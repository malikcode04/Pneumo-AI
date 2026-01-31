from fpdf import FPDF
import datetime
import tempfile
import os
from PIL import ImageDraw, ImageFont

class ClinicalReport(FPDF):
    def header(self):
        # Logo placeholder (text for now, image if we had one)
        self.set_font('Helvetica', 'B', 20)
        self.set_text_color(0, 212, 255) # Medical Blue
        self.cell(0, 10, 'Pneumo AI', align='L', new_x="LMARGIN", new_y="NEXT")
        
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, 'Advanced Clinical Decision Support', align='L', new_x="LMARGIN", new_y="NEXT")
        self.ln(5)
        
        # Line break
        self.set_draw_color(200, 200, 200)
        self.line(10, 35, 200, 35)
        self.ln(10)

    # ... (footer remains same)

# ...


def stamp_image(img, view_position="PA"):
    """Adds timestamp and view metadata watermark to image."""
    try:
        from PIL import ImageDraw, ImageFont
        
        # Ensure image is in RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Simple text watermark
        font_size = max(20, int(height * 0.03))
        try:
            # Try to use a default font
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        text = f"Pneumo AI | {timestamp} | View: {view_position}"
        
        # Draw semi-transparent background for text
        text_bbox = draw.textbbox((10, 10), text, font=font)
        draw.rectangle([text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5], fill=(0, 0, 0, 128))
        draw.text((10, 10), text, font=font, fill=(0, 212, 255))
        
        return img
    except Exception as e:
        return img

def create_pdf_report(metadata, prediction_results, images, doctor_name="Radiologist", doctor_statement=None, view_position="PA"):
    """Generates a complete clinical PDF report."""
    pdf = ClinicalReport()
    pdf.add_page()
    
    # 1. Patient Info Header
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Patient Information', new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font('Helvetica', '', 12)
    col_width = 95
    
    # Info Grid
    pdf.cell(col_width, 8, f"Patient Name: {metadata.get('Name', 'Anonymous')}")
    pdf.cell(col_width, 8, f"Patient ID: {metadata.get('Patient ID', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    
    pdf.cell(col_width, 8, f"Age / Gender: {metadata.get('Age', 'N/A')} / {metadata.get('Gender', 'N/A')}")
    pdf.cell(col_width, 8, f"Study Date: {metadata.get('Study Date', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    
    pdf.cell(col_width, 8, f"Modality: {metadata.get('Modality', 'XR')}")
    pdf.cell(col_width, 8, f"View Position: {view_position}", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(10)

    # 2. Findings Summary
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'AI Analysis Findings', new_x="LMARGIN", new_y="NEXT")
    
    # Risk Badge
    label = prediction_results.get('predicted_label', 'Unknown')
    conf = prediction_results.get('confidence', 0.0)
    
    # Analyze Specific Type
    subtype = "Unspecified"
    if "Bacterial" in label: subtype = "Bacterial Origin"
    elif "Viral" in label: subtype = "Viral Origin"
    
    pdf.set_font('Helvetica', 'B', 16)
    if 'Pneumonia' in label and conf > 0.5:
        pdf.set_text_color(220, 53, 69) # Red
        status = "CRITICAL FINDING - PNEUMONIA"
    elif label == 'Normal':
        pdf.set_text_color(40, 167, 69) # Green
        status = "NORMAL STUDY"
    else:
        pdf.set_text_color(255, 193, 7) # Orange
        status = "UNCERTAIN DIAGNOSIS"
        
    pdf.cell(0, 10, status, new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_text_color(0, 0, 0)
    
    # Detailed Findings
    pdf.set_font('Helvetica', '', 12)
    
    # Row 1: Primary Class
    pdf.cell(50, 10, "Classification:", border=0)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, f"{label}", new_x="LMARGIN", new_y="NEXT")
    
    # Row 2: Sub-Category (Visible if Pneumonia)
    if "Pneumonia" in label:
        pdf.set_font('Helvetica', '', 12)
        pdf.cell(50, 10, "Pathology Category:", border=0)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 10, f"{subtype}", new_x="LMARGIN", new_y="NEXT")
    
    # Row 3: Confidence
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(50, 10, "Model Confidence:", border=0)
    
    conf_str = f"{conf:.2%}"
    if conf > 0.9: conf_desc = "(High Accuracy)"
    elif conf > 0.7: conf_desc = "(Moderate)"
    else: conf_desc = "(Low/Uncertain)"
    
    pdf.cell(0, 10, f"{conf_str} {conf_desc}", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(10)

    # 3. Doctor's Observations (NEW)
    if doctor_statement:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, "Doctor's Final Observations", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font('Helvetica', 'I', 11)
        pdf.multi_cell(0, 6, doctor_statement)
        pdf.ln(10)

    # 4. Images
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Visual Analysis', new_x="LMARGIN", new_y="NEXT")
    
    if pdf.get_y() > 190:
        pdf.add_page()
        
    y_pos = pdf.get_y()
    img_height = 80
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_orig:
        if 'original' in images and images['original']:
             marked_img = stamp_image(images['original'], view_position)
             marked_img.save(tmp_orig.name)
             pdf.image(tmp_orig.name, x=15, y=y_pos, w=80)
             pdf.text(45, y_pos + img_height + 5, "Original X-Ray")
             
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_heat:
         if 'overlay' in images and images['overlay']:
             marked_img = stamp_image(images['overlay'], view_position)
             marked_img.save(tmp_heat.name)
             pdf.image(tmp_heat.name, x=115, y=y_pos, w=80)
             pdf.text(140, y_pos + img_height + 5, "AI Evidence Map")

    pdf.set_y(y_pos + img_height + 20)

    # 5. Clinical Metrics
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Detailed Metrics', new_x="LMARGIN", new_y="NEXT")
    
    metrics = prediction_results.get('clinical_metrics', {})
    uncertainty = prediction_results.get('uncertainty', {})
    
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 6, f"Sensitivity: {metrics.get('sensitivity', 'N/A')}\n"
                         f"Specificity: {metrics.get('specificity', 'N/A')}\n"
                         f"Uncertainty: {uncertainty.get('entropy', 0):.3f}")
    
    # 6. Sign-off
    pdf.ln(10)
    if pdf.get_y() > 250: # Avoid breaking signature page awkwardly
        pdf.add_page()
        
    pdf.set_draw_color(0, 0, 0)
    pdf.line(10, pdf.get_y(), 80, pdf.get_y()) # Signature line
    pdf.ln(2)
    
    pdf.set_font('Helvetica', 'I', 10)
    pdf.cell(0, 10, f"Electronically Signed by: Dr. {doctor_name.capitalize()}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Remove temporary files
    try:
        os.unlink(tmp_orig.name)
        os.unlink(tmp_heat.name)
    except:
        pass

    return bytes(pdf.output())

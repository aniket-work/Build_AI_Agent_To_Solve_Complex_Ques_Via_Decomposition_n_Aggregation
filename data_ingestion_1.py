import json
import logging
from fpdf import FPDF
from constants import REPORTS_FOLDER, PDF_FONT, PDF_FONT_SIZE, PDF_TITLE_ALIGN, PDF_CONTENT_MARGIN

logger = logging.getLogger(__name__)

def load_config():
    with open('config.json', 'r') as config_file:
        return json.load(config_file)

def create_detailed_pdf(year, details, path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font(PDF_FONT, size=PDF_FONT_SIZE)
    pdf.cell(200, 10, txt=details["title"], ln=True, align=PDF_TITLE_ALIGN)
    pdf.ln(PDF_CONTENT_MARGIN)
    pdf.multi_cell(0, 10, txt=details["content"])
    pdf.output(path)
    logger.debug(f"Created PDF: {path}")

async def generate_pdfs():
    config = load_config()
    detailed_reports = config['detailed_reports']

    for year, details in detailed_reports.items():
        file_path = f"{REPORTS_FOLDER}{year}_report.pdf"
        create_detailed_pdf(year, details, file_path)
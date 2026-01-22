# app.py - MediAssist Section-Aware Medical Report Explainer

import gradio as gr
import pdfplumber
from docx import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ---------------------------
# 1️⃣ Load AI Model
# ---------------------------
MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ---------------------------
# 2️⃣ File text extraction
# ---------------------------
def extract_text(file):
    if file.name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file.name) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    elif file.name.endswith(".docx"):
        doc = Document(file.name)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])
        return text
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return "Unsupported file format. Please upload PDF, DOCX, or TXT."

# ---------------------------
# 3️⃣ Section splitting
# ---------------------------
def split_sections(text):
    headers = ["diagnosis", "lab results", "prescriptions", "imaging", "medications", "history", "recommendations", "notes"]
    sections = {}
    current_header = "General"
    sections[current_header] = ""

    for line in text.splitlines():
        line_lower = line.strip().lower()
        matched_header = next((h for h in headers if h in line_lower), None)
        if matched_header:
            current_header = matched_header.capitalize()
            sections[current_header] = ""
        else:
            sections[current_header] += line + "\n"
    return sections

# ---------------------------
# 4️⃣ AI explanation
# ---------------------------
def generate_explanation(section_text, question):
    prompt = f"Medical report section: '''{section_text}'''\n\nQuestion: {question}\nAnswer in patient-friendly language without giving medical advice:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ---------------------------
# 5️⃣ Gradio interface
# ---------------------------
def mediassist(file, question, section_choice):
    report_text = extract_text(file)
    if "Unsupported" in report_text:
        return report_text

    sections = split_sections(report_text)

    if section_choice == "All Sections":
        combined_text = "\n".join([f"{k}: {v}" for k, v in sections.items()])
        answer = generate_explanation(combined_text, question)
    else:
        section_text = sections.get(section_choice, "")
        if not section_text.strip():
            return f"No content found in section '{section_choice}'"
        answer = generate_explanation(section_text, question)
    return answer

# ---------------------------
# 6️⃣ Section options
# ---------------------------
section_options = ["All Sections", "Diagnosis", "Lab results", "Prescriptions", "Imaging", "Medications", "History", "Recommendations", "Notes"]

# ---------------------------
# 7️⃣ Launch Gradio app
# ---------------------------
interface = gr.Interface(
    fn=mediassist,
    inputs=[
        gr.File(label="Upload Medical Report (PDF, DOCX, TXT)"),
        gr.Textbox(label="Ask a question about the report", placeholder="e.g., Explain the diagnosis section"),
        gr.Dropdown(section_options, label="Choose section to query", value="All Sections")
    ],
    outputs=gr.Textbox(label="AI Explanation"),
    title="MediAssist - Section-Aware Medical Report Explainer",
    description="Upload a medical report and ask questions about specific sections or the entire report. AI provides patient-friendly explanations."
)

if __name__ == "__main__":
    interface.launch()


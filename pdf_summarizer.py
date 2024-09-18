import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from io import BytesIO

# Helper functions
def initialize_llm(api_key, model="Llama3-8b-8192"):
    """
    Initializes the LLM with the given API key and model.
    """
    return ChatGroq(api_key=api_key, model=model)

def extract_text_from_pdf(pdf_file_bytes):
    """
    Extracts text from an in-memory PDF file using PyMuPDF (fitz).
    
    Args:
        pdf_file_bytes (BytesIO): In-memory PDF file content.
        
    Returns:
        str: The extracted text from the PDF.
    """
    doc = fitz.open(stream=pdf_file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits extracted text into smaller chunks using RecursiveCharacterTextSplitter.
    
    Args:
        text (str): The extracted text from the PDF.
        chunk_size (int): The size of each text chunk.
        chunk_overlap (int): The overlap between chunks.
        
    Returns:
        list: A list of split documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.create_documents([text])

def create_prompts():
    """
    Creates the map and final prompt templates.
    
    Returns:
        tuple: A tuple containing the map and final prompt templates.
    """
    chunks_prompt = """
    Please summarize the below documents:
    documents:`{text}'
    Summary:
    """
    map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)
    
    final_prompt = '''
    Provide the final summary of the entire documents with these important points.
    Add a Title, Start the precise summary with an introduction and provide the summary in numbered 
    points for the documents.
    documents:{text}
    '''
    final_prompt_template = PromptTemplate(input_variables=['text'], template=final_prompt)
    
    return map_prompt_template, final_prompt_template

def create_summary_chain(llm, map_prompt_template, final_prompt_template):
    """
    Creates a summary chain with map-reduce structure.
    
    Args:
        llm: The initialized language model.
        map_prompt_template: The map prompt template.
        final_prompt_template: The final combine prompt template.
        
    Returns:
        Chain: A summarization chain.
    """
    return load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=final_prompt_template,
        verbose=True,
    )

def summarize_pdf(api_key, pdf_file_bytes):
    """
    Summarizes the content of a PDF document using a map-reduce chain.
    
    Args:
        api_key (str): The API key for the language model.
        pdf_file_bytes (BytesIO): In-memory PDF file.
        
    Returns:
        str: The summary output.
    """
    # Initialize the LLM
    llm = initialize_llm(api_key)
    
    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(pdf_file_bytes)
    
    # Split the extracted text into smaller chunks
    final_documents = split_text_into_chunks(extracted_text)
    
    # Create prompts
    map_prompt_template, final_prompt_template = create_prompts()
    
    # Create the summary chain
    summary_chain = create_summary_chain(llm, map_prompt_template, final_prompt_template)
    
    # Run the summary chain on the documents
    return summary_chain.run(final_documents)

# Streamlit UI
st.title("PDF Summarizer")

# Sidebar for API Key
st.sidebar.header("Enter API Key")
api_key = st.sidebar.text_input("GROQ API Key", type="password")

# Main page for file upload
st.header("Upload your PDF document")
pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

# Display a button for processing
if st.button("Summarize PDF"):
    if not api_key:
        st.error("Please enter the API key in the sidebar.")
    elif not pdf_file:
        st.error("Please upload a PDF file.")
    else:
        # Process and summarize the PDF
        with st.spinner("Summarizing your document..."):
            try:
                # Pass the uploaded file's bytes content to the summarize function
                pdf_file_bytes = BytesIO(pdf_file.read())
                summary = summarize_pdf(api_key, pdf_file_bytes)
                st.success("Summarization complete!")
                st.write("### Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"An error occurred: {e}")

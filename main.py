from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain


def load_env_variables():
    load_dotenv()
    return os.getenv("GROQ_API_KEY")


def initialize_llm(api_key, model="Llama3-8b-8192"):
    return ChatGroq(api_key=api_key, model=model)


def load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)


def create_prompts():
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
    return load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=final_prompt_template,
        # verbose=True,
    )


def summarize_pdf(pdf_path):
    api_key = load_env_variables()
    llm = initialize_llm(api_key)
    map_prompt_template, final_prompt_template = create_prompts()
    
    # Load and split the document
    final_documents = load_and_split_pdf(pdf_path)
    
    # Create the summary chain
    summary_chain = create_summary_chain(llm, map_prompt_template, final_prompt_template)
    
    # Run the summary chain on the documents
    return summary_chain.invoke(final_documents)


if __name__ == "__main__":
    pdf_path = "research/RFP.pdf"  
    summary = summarize_pdf(pdf_path)
    print(summary['output_text'])




import streamlit as st
import validators
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader


def initialize_llm(api_key, model="Gemma-7b-It"):
    return ChatGroq(api_key=api_key, model=model)


def extract_text(generic_url):
    if "youtube.com" in generic_url:
        loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
    else:
        loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
    text=loader.load()
    return text


def create_prompt():
    prompt_template="""
    Provide a summary of the following content in 300 words:
    Content:{text}

    """
    prompt=PromptTemplate(template=prompt_template,input_variables=["text"])
    return prompt


def create_summary_chain(llm, prompt):
    return load_summarize_chain(llm=llm,chain_type="stuff",prompt=prompt)


def summarize(api_key, generic_url):
    llm = initialize_llm(api_key)
    extracted_text = extract_text(generic_url)
    prompt = create_prompt()
    summary_chain = create_summary_chain(llm, prompt)
    return summary_chain.invoke(extracted_text)


# Streamlit UI
st.title("YouTube video Summarizer")

# Sidebar for API Key
st.sidebar.header("Enter API Key")
api_key = st.sidebar.text_input("GROQ API Key", type="password")

# Sub headed
generic_url=st.text_input('Pls enter URL URL', placeholder="URL")

# Display a button for processing
if st.button("Summarize"):
    if not api_key:
        st.error("Please enter the API key in the sidebar.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")
    else:
        # Process and summarize the PDF
        with st.spinner("Summarizing your document..."):
            try:
                summary = summarize(api_key, generic_url)
                st.success("Summarization complete!")
                st.write("### Summary:")
                st.write(summary['output_text'])
            except Exception as e:
                st.error(f"An error occurred: {e}")



# if __name__ == "__main__":
#     url = "https://www.youtube.com/watch?v=wFdFLWc-W4k&t=1589s"  
#     summary = summarize(api_key="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",generic_url=url)
#     print(summary)



import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# from hosted_model_call import predict_custom_trained_model_sample
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema import Document
from scraper_cases import cases_keyword_search_scrape_bailii

from dotenv import load_dotenv
load_dotenv()

# Set environment variables
os.environ['OPENAI_API_KEY'] =  os.getenv('OPENAI_API_KEY')



formulate_response_prompt = ChatPromptTemplate.from_template(
"""
You are an AI assistant tasked with reviewing the query provided and formulating a response based on the information provided. Your primary goal is to provide a concise and informative response that addresses the query effectively.
You should aim to provide a clear and accurate answer to the question asked, avoiding unnecessary details or overly complex explanations.
AND SHOULD ONLY SUMMARIZE THE INFORMATION IN THE DOCUMENTS, NOT PROVIDE A DIRECT ANSWER.
Question asked: {input}

Context (Use only this information to formulate your response):
<context>
{context}
</context>
Also display links of documents sumamrized at the bottom of the response.
""")


def add_case_data_to_vectorstore(case_data, vectorstore_path="openai_vectorstore"):
    # Initialize OpenAI client
    client = OpenAI()
    
    # Create OpenAIEmbeddings instance
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Convert case_data to Document objects
    documents = [
        Document(
            page_content=case['content'],
            metadata={'case_id': case['case_id'], 'url': case['url']}
        )
        for case in case_data
    ]
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    
    print(f"Number of document splits created: {len(splits)}")
    
    # Check if vectorstore exists and load it, or create a new one
    if os.path.exists(vectorstore_path):
        print("Loading existing vectorstore...")
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        print("Adding new documents to existing vectorstore...")
        vectorstore.add_documents(splits)
    else:
        print("Creating new vectorstore...")
        vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Save the updated vectorstore
    vectorstore.save_local(vectorstore_path)
    print(f"Vectorstore saved to {vectorstore_path}")
    
    return vectorstore

def get_vectorstore(vectorstore_path="openai_vectorstore"):
    if not os.path.exists(vectorstore_path):
        print(f"Vectorstore not found at {vectorstore_path}")
        return None
    
    client = OpenAI()
    embeddings = OpenAIEmbeddings(client=client)
    
    try:
        vectorstore = FAISS.load_local(vectorstore_path, embeddings)
        print("Vectorstore loaded successfully.")
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None

def create_custom_retrieval_chain(vectorstore):
    # Initialize the LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini") 
    
    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, formulate_response_prompt)
    
    # Create the retriever
    retriever = vectorstore.as_retriever()
    
    # Create and return the retrieval chain
    return create_retrieval_chain(retriever, document_chain)

def extract_urls(response):
    urls = []
    for doc in response["context"]:
        if "url" in doc.metadata:
            urls.append(doc.metadata["url"])
    return list(set(urls))  # Remove duplicates


def get_response(vectorstore, user_prompt):
    # Create the retrieval chain
    retrieval_chain = create_custom_retrieval_chain(vectorstore)
    
    # Get the response
    response = retrieval_chain.invoke({
        "input": user_prompt,
    })


    return {"answer":response["answer"], "links":extract_urls(response)}



def search_cases(query):
    # Initialize the RAG system

    # Initialize the OpenAI model
    model = ChatOpenAI(model_name="gpt-4o-mini")  # Use the appropriate model name

    # Create the prompt template
    get_keyword_prompt = ChatPromptTemplate.from_template(
    """
    You are an AI assistant tasked with reviewing the query provided and breaking it down into keywords that can be searched in a legal database. Your primary goal is to identify the main keywords that are relevant to the query.
    ONLY provide keywords that are directly related to the query and avoid including any unnecessary or irrelevant terms.
    DO NOT PROVIDE A FULL SENTENCE RESPONSE. Instead, list the keywords separated by commas.

    Question asked: {input}
    """)

    # Create an output parser
    output_parser = CommaSeparatedListOutputParser()

    # Create the chain
    chain = get_keyword_prompt | model | output_parser

    # Function to get keywords
    def get_keywords(query):
        keywords = chain.invoke({"input": query})
        return keywords

    # Example usage
    # query = "summary of the cases of negligence in the workplace"
    keywords = get_keywords(query)
    list_html=cases_keyword_search_scrape_bailii(keywords)
    # Load the vectorstore                
    print(list_html)
    vectorstore = add_case_data_to_vectorstore(list_html)
    return get_response(vectorstore, query)

if __name__ == "__main__":
    main()
    # prompting_ui()

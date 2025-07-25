
# Install necessary packages (run once)
!pip install langchain langchain-openai openai qdrant-client pypdf python-docx faiss-cpu

import os
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Step 1: Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"  # replace with your key

# Step 2: Upload your file (use Google Colab's upload widget)
from google.colab import files
uploaded = files.upload()

# Step 3: Automatically get the uploaded file path
uploaded_file_path = next(iter(uploaded))  # gets the first uploaded file name
print(f"Uploaded file: {uploaded_file_path}")

# Step 4: Load documents depending on file type
file_ext = uploaded_file_path.split('.')[-1].lower()
if file_ext == 'pdf':
    loader = UnstructuredPDFLoader(uploaded_file_path)
elif file_ext in ['docx', 'doc']:
    loader = UnstructuredWordDocumentLoader(uploaded_file_path)
else:
    raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")

documents = loader.load()

# Step 5: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Step 6: Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(texts, embeddings)

# Step 7: Create RetrievalQA chain
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = OpenAI(temperature=0)  # temperature=0 for deterministic answers

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

# Step 8: Ask a question and print the answer + sources
query = "Tell me about Mubadala's subsidiaries and their main activities."

response = qa_chain.invoke({"query": query})  # use invoke() instead of deprecated call

print("\nAnswer:")
print(response.get('result', 'No result found.'))

print("\nSource Documents:")
source_docs = response.get('source_documents', [])
if source_docs:
    for doc in source_docs:
        source = doc.metadata.get('source', 'No source')
        snippet = doc.page_content[:200].replace('\n', ' ') + '...'
        print(f"- Source: {source}\n  Snippet: {snippet}\n")
else:
    print("No source documents available.")

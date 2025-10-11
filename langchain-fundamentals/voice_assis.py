import streamlit as st
import whisper
import sounddevice as sd
import soundfile as sf
from elevenlabs import ElevenLabs, save
import hashlib
# from elevenlabs import generate, save



from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

import tempfile
from dotenv import load_dotenv
import os
from typing import List
from langchain_core.documents import Document

load_dotenv()

# class DocumentProcessor:
#     def __init__(self):
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
#         )
#         self.embeddings = OpenAIEmbeddings()

#     def load_documents(self, directory: str) -> List[Document]:
#         """Load documents from different file types"""
#         loaders = {
#             ".pdf": DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
#             ".txt": DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
#             ".md": DirectoryLoader(
#                 directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
#             ),
#         }

#         documents = []
#         for file_type, loader in loaders.items():
#             try:
#                 documents.extend(loader.load())
#                 print(f"Loaded {file_type} documents")
#             except Exception as e:
#                 print(f"Error loading {file_type} documents: {str(e)}")

#         return documents

#     def process_documents(self, documents: List[Document]) -> List[Document]:
#         """Split documents into chunks"""
#         return self.text_splitter.split_documents(documents)

#     def create_vector_store(
#         self, documents: List[Document], persist_directory: str
#     ) -> Chroma:
#         """Create and persist vector store if it doesn't exist, otherwise load existing one"""
#         # Check if persist_directory exists and has content
#         if os.path.exists(persist_directory) and os.listdir(persist_directory):
#             print(f"Loading existing vector store from {persist_directory}")
#             # Load existing vector store
#             vector_store = Chroma(
#                 persist_directory=persist_directory, embedding_function=self.embeddings
#             )
#             print("i got here in the alreadyexisting")
#             # Append new documents if provided
#             if documents:
#                 vector_store.add_documents(documents)
#                 vector_store.persist()
#                 print(f"Added {len(documents)} new docs to existing store")

#         else:
#             print(f"Creating new vector store in {persist_directory}")
#             # Create directory if it doesn't exist
#             os.makedirs(persist_directory, exist_ok=True)

#             # Create new vector store
#             vector_store = Chroma.from_documents(
#                 documents=documents,
#                 embedding=self.embeddings,
#                 persist_directory=persist_directory,
#             )
#             vector_store.persist()
#             print("i got here in the not alreadyexisting")

#         return vector_store

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.embeddings = OpenAIEmbeddings()

    def _hash_content(self, text: str) -> str:
        """Generate a stable hash for deduplication"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def load_documents(self, directory: str) -> List[Document]:
        """Load documents from different file types"""
        loaders = {
            ".pdf": DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".txt": DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
            ".md": DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        }

        documents = []
        for file_type, loader in loaders.items():
            try:
                docs = loader.load()
                for doc in docs:
                    # add a hash ID so we can deduplicate later
                    doc.metadata["doc_id"] = self._hash_content(doc.page_content)
                documents.extend(docs)
                print(f"Loaded {file_type} documents")
            except Exception as e:
                print(f"Error loading {file_type} documents: {str(e)}")

        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        split_docs = self.text_splitter.split_documents(documents)
        for doc in split_docs:
            if "doc_id" not in doc.metadata:
                doc.metadata["doc_id"] = self._hash_content(doc.page_content)
        return split_docs

    def create_vector_store(self, documents: List[Document], persist_directory: str) -> Chroma:
        """Create or update vector store, avoiding duplicate docs."""

        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            print(f"Loading existing vector store from {persist_directory}")
            vector_store = Chroma(
                persist_directory=persist_directory, embedding_function=self.embeddings
            )
            print("i got here in the alreadyexisting")

            if documents:
                # get all existing doc_ids
                existing_ids = set()
                try:
                    existing = vector_store.get(include=["metadatas"])
                    for md in existing["metadatas"]:
                        if "doc_id" in md:
                            existing_ids.add(md["doc_id"])
                except Exception as e:
                    print(f"Error fetching metadata: {e}")

                # only keep docs with new IDs
                new_docs = [doc for doc in documents if doc.metadata.get("doc_id") not in existing_ids]

                if new_docs:
                    vector_store.add_documents(new_docs)
                    vector_store.persist()
                    print(f"Added {len(new_docs)} new docs to existing store")
                else:
                    print("No new docs to add (all duplicates)")

        else:
            print(f"Creating new vector store in {persist_directory}")
            os.makedirs(persist_directory, exist_ok=True)

            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory,
            )
            vector_store.persist()
            print(f"Created new vector store with {len(documents)} docs")

        return vector_store
        

    

class VoiceGenerator:
    def __init__(self, api_key):
        self.client = ElevenLabs(api_key=api_key)

        # Fetch voices from ElevenLabs API once
        try:
            voices = self.client.voices.get_all()

            # Map names -> ids
            self.available_voices = {v.name: v.voice_id for v in voices.voices}

        except Exception as e:
            print(f"Error fetching voices: {e}")
            # fallback hardcoded mapping (not recommended)
            self.available_voices = {}

        self.default_voice = "Rachel"

        
    def generate_voice_response(self, text: str, voice_name: str = None) -> str:
        """Generate voice response using ElevenLabs SDK"""
        if not text.strip():
            print("No text provided for TTS")
            return None

        try:
            # Pick correct voice_id
            if voice_name and voice_name in self.available_voices:
                voice_id = self.available_voices[voice_name]
            else:
                voice_id = self.available_voices.get(self.default_voice)

            if not voice_id:
                raise ValueError("No valid voice_id found")

            # Generate audio
            audio = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128"
            )

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                save(audio, temp_audio.name)
                return temp_audio.name

        except Exception as e:
            print(f"Error generating voice response: {e}")
            return None


        
class VoiceAssistantRAG:
    def __init__(self, elevenlabs_api_key):
        self.whisper_model = whisper.load_model("base")
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.qa_chain = None
        self.sample_rate = 44100
        self.voice_generator = VoiceGenerator(elevenlabs_api_key)

    def setup_vector_store(self, vector_store):
        """Initialize the vector store and QA chain"""
        if vector_store is None:
            raise ValueError("Vector store is None. Did you forget to create it?")
        self.vector_store = vector_store

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=memory,
            verbose=True,
        )

    def record_audio(self, duration=5):
        """Record audio from microphone"""
        recording = sd.rec(
            int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1
        )
        sd.wait()
        return recording
    
    def transcribe_audio(self, audio_array):
        """Transcribe audio using Whisper"""
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf.write(temp_audio.name, audio_array, self.sample_rate)
            temp_audio_path = temp_audio.name
            temp_audio.close()  # Fecha o arquivo antes de transcrever
            result = self.whisper_model.transcribe(temp_audio_path)
        finally:
            try:
                os.unlink(temp_audio_path)
            except PermissionError:
                print(f"Não foi possível deletar o arquivo temporário: {temp_audio_path}")
        return result["text"]
    

    def generate_response(self, query):
        """Generate response using RAG system"""
        if self.qa_chain is None:
            return "Error: Vector store not initialized"

        response = self.qa_chain.invoke({"question": query})
        return response["answer"]

    def text_to_speech(self, text: str, voice_name: str = None) -> str:
        """Convert text to speech"""
        return self.voice_generator.generate_voice_response(text, voice_name)
    
    
def setup_knowledge_base():
    st.title("Knowledge Base Setup")

    doc_processor = DocumentProcessor()

    uploaded_files = st.file_uploader(
        "Upload your documents", accept_multiple_files=True, type=["pdf", "txt", "md"]
    )

    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            temp_dir = tempfile.mkdtemp()
            print(temp_dir)

            # Save uploaded files
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                print(file_path)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            try:
                # Process documents
                documents = doc_processor.load_documents(temp_dir)
                processed_docs = doc_processor.process_documents(documents)

                # Create vector store
                vector_store = doc_processor.create_vector_store(
                    processed_docs, "knowledge_base"
                )

                # Store in session state
                st.session_state.vector_store = vector_store

                st.success(f"Processed {len(processed_docs)} document chunks!")

            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
            finally:
                # Cleanup
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
                
def main():
    st.set_page_config(page_title="Voice RAG Assistant", layout="wide")

    # Check for API keys
    elevenlabs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not all([elevenlabs_api_key, openai_api_key]):
        st.error(
            "Please set ELEVEN_LABS_API_KEY and OPENAI_API_KEY in your environment variables"
        )
        return

    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Setup Knowledge Base", "Voice Assistant"])

    if page == "Setup Knowledge Base":
        vector_store = setup_knowledge_base()
        if vector_store:
            st.session_state.vector_store = vector_store

    else:  # Voice Assistant page
        if "vector_store" not in st.session_state:
            st.error("Please setup knowledge base first!")
            return

        st.title("Voice Assistant RAG System")

        # Initialize assistant
        assistant = VoiceAssistantRAG(elevenlabs_api_key)
        # Initialize the vector store and QA chain
        assistant.setup_vector_store(st.session_state.vector_store)

        # Voice selection
        try:
            available_voices = list(assistant.voice_generator.available_voices.keys())
            print(available_voices)
            if available_voices:
                selected_voice = st.sidebar.selectbox(
                    "Select Voice",
                    available_voices,
                    index=(
                        available_voices.index("Rachel")
                        if "Rachel" in available_voices
                        else 0
                    ),
                )
            else:
                st.warning("No voices available. Using default voice.")
                selected_voice = "Rachel"
        except Exception as e:
            st.error(f"Error loading voices: {e}")
            selected_voice = "Rachel"

        # Recording duration
        duration = st.sidebar.slider("Recording Duration (seconds)", 1, 10, 5)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Start Recording"):
                with st.spinner(f"Recording for {duration} seconds..."):
                    audio_data = assistant.record_audio(duration)
                    st.session_state.audio_data = audio_data
                    st.success("Recording completed!")

        with col2:
            if st.button("Process Recording"):
                if "audio_data" not in st.session_state:
                    st.error("Please record audio first!")
                    return

                # Process recording
                with st.spinner("Transcribing..."):
                    query = assistant.transcribe_audio(st.session_state.audio_data)
                    st.write("You said:", query)

                with st.spinner("Generating response..."):
                    try:
                        response = assistant.generate_response(query)
                        st.write("Response:", response)
                        st.session_state.last_response = response
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        return

                with st.spinner("Converting to speech..."):
                    audio_file = assistant.voice_generator.generate_voice_response(
                        response, selected_voice
                    )
                    if audio_file:
                        st.audio(audio_file)
                        os.unlink(audio_file)
                    else:
                        st.error("Failed to generate voice response")

        # Display chat history
        if "chat_history" in st.session_state:
            st.subheader("Chat History")
            for q, a in st.session_state.chat_history:
                st.write("Q:", q)
                st.write("A:", a)
                st.write("---")


if __name__ == "__main__":
    main()
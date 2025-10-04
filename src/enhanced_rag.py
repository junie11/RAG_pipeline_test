from langchain.text_splitter import RecursiveCharacterTextSplitter

class EnhancedRAG:
    def __init__(self, chunkSize, overlap):
        self.all_chunks = []
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=overlap)

    def chunkData(self, text):
        chunks = self.text_splitter.split_text(text)
        self.all_chunks.extend(chunks)
    
    def addChunks(self, chunks):
        self.all_chunks.extend(chunks)

    def splitTexts(self, text):
        return self.text_splitter.split_text(text)

    def getChunkLength(self):
        return len(self.all_chunks)
    
    def getChunks(self):
        return self.all_chunks

    # Claude suggested I add a confidence calculation method. See Appendix P. 
    def calculate_confidence(self, distance_score, threshold=0.5):
        """Calculate confidence based on distance score from Milvus"""
        # Convert distance to similarity (lower distance = higher similarity)
        similarity = 1 / (1 + distance_score)
        
        confidence = {
            'distance': distance_score,
            'similarity': similarity,
            'confidence_level': 'high' if distance_score < threshold else 'low'
        }
        return confidence
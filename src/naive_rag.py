import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from collections import Counter

class NaiveRAG:
    def __init__(self, embedding_model_name, model_name, database_name, collection_name):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("Embedding model loaded.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded.")
        self.schema = MilvusClient.create_schema(auto_id=False,
        enable_dynamic_field=False,)
        print("Schema created.")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Seq2Seq model loaded.")
        self.client = MilvusClient(database_name)
        print("Milvus client connected.")
        self.collection_name = collection_name
        # Claude recommended to drop collection if it exists (fresh start). See Appendix O. 
        if self.client.has_collection(collection_name=self.collection_name):
            print(f"Dropping existing collection '{self.collection_name}'...")
            self.client.drop_collection(collection_name=self.collection_name)
            print("Collection dropped.")
        self.queries_list = []
        self.flatten_answer = []
        self.confidence_scores = [] 
        # Claude code recommended I add a contexts list and additional search method adjustments. See Appendix R.
        self.contexts_list = []


    
    def create_dataBase(self, corpus, corpus_id, embeddings, dim):
        id_ = corpus.index.tolist()
        embedding = embeddings.tolist()
        print("Columns defined.")
        self.schema.add_field(field_name="id_", datatype=DataType.INT64, is_primary=True)
        self.schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        self.schema.add_field(field_name=corpus_id, datatype=DataType.VARCHAR, max_length=1000)
        print("Fields added to schema.")
        self.client.create_collection(collection_name=self.collection_name, dimension=dim)
        print("Collection created.")
        rag_data = [{"id" : id_[i], corpus_id: corpus[corpus_id].iloc[i], "vector": embedding[i]} for i in range(len(embeddings))]
        self.client.insert(collection_name=self.collection_name, data=rag_data)
        print("Data inserted successfully.")
        index_params = MilvusClient.prepare_index_params()

        # Add an index on the embedding field
        index_params.add_index(field_name="vector", index_type="IVF_FLAT", params={"nlist": 128})

        # Create the index
        try:
            self.client.create_index(collection_name=self.collection_name, index_params=index_params)
        except Exception as e:
            print(f"Index creation result: {e}")
        print("Index created successfully.")
        # Load collection into memory (required for search)
        self.client.load_collection(collection_name=self.collection_name)
        print("Collection loaded into memory")
    
    # Claude recommended I incorporate confidence score code in this module. See Appendix Q.
    def search(self, query, top_k, corpus, prompt, use_enhanced=False, enhancedRag=None):
        # Step 1: Embed the query
        new_query_embedding = self.embedding_model.encode([query])
        # Perform initial search to get top passages
        new_context = self.client.search(collection_name=self.collection_name, data=new_query_embedding, limit=top_k)

        if use_enhanced:
            # Enhanced RAG with confidence scoring on top-k
            top_results = new_context[0][:top_k]
            # Calculate confidence for each result
            high_conf_passages = []
            all_confidences = []
            for result in top_results:
                result_id = result['id']
                distance = result['distance']
                passage = corpus[result_id]
                confidence = enhancedRag.calculate_confidence(distance)

                if confidence['confidence_level'] == 'high':
                    high_conf_passages.append(passage)
                all_confidences.append(confidence)
            # Combine multiple passages or fallback to top-1
            if len(high_conf_passages) > 0:
                top_passage = " ".join(high_conf_passages[:2])  # Use top 2 high-conf passages
            else:
                top_passage = corpus[top_results[0]['id']]
            # Store aggregate confidence
            self.confidence_scores.append(all_confidences[0])
        else:    
            top_result_id = new_context[0][0]['id']  # Get the ID of top result
            top_passage = corpus[top_result_id]  # Get actual passage text
        # Store context for RAGAs evaluation
        self.contexts_list.append([top_passage])  # RAGAs expects list of contexts

        # Step 3: Generate answer using the retrieved passage and the query
        new_prompt = f"""{prompt} \n Context: {top_passage}: \n Question: {query} """
        all_queries_input = self.tokenizer(new_prompt, return_tensors="pt", truncation=True, max_length=512)
        all_queries_output = self.model.generate(**all_queries_input)
        queries_answer = self.tokenizer.batch_decode(all_queries_output, skip_special_tokens=True)

        # Step 4: Store the answer in the list
        self.queries_list.append(queries_answer)

    def sanityCheck(self, collection_name):
        print("Entity count:", self.client.get_collection_stats(collection_name)["row_count"])
        print("Collection schema:", self.client.describe_collection(collection_name))

    # I asked Claude to help me write this function. See Appendix L. 
    def calculateEM(self, references, limit=100):
       """Calculate Exact Match (EM) score between generated answers and ground truths."""
       counter = 0
       self.flatten_answer = [actual_answer[0] for actual_answer in self.queries_list]

       for index, truth in enumerate(references):
           if index >= limit:
               break
           if index < len(self.flatten_answer) and truth == self.flatten_answer[index]:
               counter += 1

       em_score = counter / min(limit, len(self.flatten_answer), len(references))
       print(f"EM Score: {counter}/{min(limit, len(self.flatten_answer), len(references))} = {em_score:.4f}")
       return em_score

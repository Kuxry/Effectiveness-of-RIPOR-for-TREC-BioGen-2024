# Effectiveness of RIPOR for TREC BioGen(QA) in 2024

Recent research has shown that transformer networks can be used as differentiable search indexes by representing each document as a sequence of document ID tokens. These generative retrieval models cast the retrieval problem into a document ID generation problem for each given query. Despite their elegant design, existing generative retrieval models often underperform when applied to large-scale, real-world collections, which has raised concerns about their practical applicability.

# Explanation of the RIPOR Model

The **RIPOR (Relevance Information Propagation for Optimal Retrieval)** model is introduced in the paper *"Scalable and Effective Generative Information Retrieval."* This model represents an advanced approach to generative retrieval, which transforms the retrieval problem into a document ID generation task. RIPOR leverages transformer networks to treat each document as a sequence of tokens, generating document IDs based on query input.

### Key Features of RIPOR:
1. **Scalability**: Unlike traditional generative retrieval models that struggle with large-scale datasets, RIPOR is designed to handle massive collections efficiently, making it highly scalable for practical, real-world applications.
   
2. **Efficiency**: The model integrates beam search techniques during the retrieval process, which enhances performance by improving the selection of relevant document IDs from the initial stages of decoding.
   
3. **Relevance Propagation**: One of RIPOR's key innovations is its ability to propagate relevance information throughout the retrieval process. This helps in narrowing down the most relevant documents based on the query, optimizing retrieval performance.

### RIPOR's Limitation:
Despite these advantages, RIPOR faces a critical limitation: it can only retrieve documents relevant to a given query. It cannot synthesize the retrieved documents to form a cohesive and complete answer, particularly in scenarios where the query asks for a structured, combined response. My current research aims to address this shortcoming by enhancing RIPOR's capabilities to generate complete answers based on multiple relevant documents.


# My work on the TREC 2024 BioGen task
In my work on the TREC 2024 BioGen task, I will use the RIPOR model to assess the feasibility of generative retrieval models in handling large-scale datasets. The 2023 PubMed dataset will be utilized to test the RIPOR model's effectiveness. RIPOR introduces several advancements in retrieval tasks, particularly by leveraging transformer networks for efficient document retrieval. However, a significant limitation of the RIPOR model is its inability to combine the most relevant documents into a complete answer based on the query’s question. It can only retrieve relevant documents but does not synthesize or structure them into a cohesive response.

To address this limitation, my current work focuses on enhancing the RIPOR model’s capabilities by developing a method that not only retrieves relevant documents but also selects and combines the most relevant ones to provide a comprehensive answer to the query.

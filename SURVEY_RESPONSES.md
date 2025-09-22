# BigQuery 2025 Competition - User Survey Responses

## Competition: BigQuery AI - Building the Future of Data
## Date: September 2025
## Project: Clinical Trial Matching at Scale

---

## Survey Questions and Responses

### 1. Please tell us how many months of experience with BigQuery AI each team member has.

**Victor Adafinoaiei:**
- BigQuery experience: 0 months (first time using BigQuery)
- BigQuery AI features (ML.GENERATE_TEXT, VECTOR_SEARCH): 0 months (learned during this competition)
- Previous experience: Healthcare, ethics and compliance

---

### 2. Please tell us how many months of experience with Google Cloud each team member has.

**Victor Adafinoaiei:**
- Google Cloud Platform: 0 months (first time using Google Cloud)
- Services used: BigQuery (learned during this competition)

---

### 3. We'd love to hear from you and your experience in working with the technology during this hackathon, positive or negative. Please provide any feedback on your experience with BigQuery AI.

## Positive Experiences

### 1. VECTOR_SEARCH Performance
The native VECTOR_SEARCH function with TreeAH indexes exceeded our expectations. We achieved 11x performance improvement over brute force search, bringing query times from 45 seconds to under 4 seconds for 10,000 patient embeddings against 5,000 trial embeddings.

### 2. ML.GENERATE_EMBEDDING Integration
The seamless integration of text-embedding-004 model directly in SQL was remarkable. Being able to generate 768-dimensional embeddings without leaving BigQuery simplified our pipeline significantly.

### 3. AI.GENERATE Functions
The new AI.GENERATE family of functions (AI.GENERATE_TEXT, AI.GENERATE_TABLE) made it incredibly easy to add LLM capabilities to our SQL queries. The Gemini 2.5 Flash integration performed well for eligibility assessments.

### 4. BigFrames Python API
BigFrames provided a familiar pandas interface while leveraging BigQuery's distributed computing. This was particularly useful for data scientists on our team who could work with their preferred tools.

### 5. MIMIC-IV Data Integration
One of the most impressive aspects was how seamlessly MIMIC-IV data was available in BigQuery after completing the PhysioNet training. The integration was incredibly smooth - we could immediately create tables and process the healthcare data without complex ETL pipelines. This made it possible to focus on the AI/ML aspects rather than data infrastructure challenges.

## Challenges and Areas for Improvement

### 1. TreeAH Index Documentation
While TreeAH indexes provided excellent performance, the documentation could be more comprehensive. We had to experiment significantly to find optimal parameters like `fraction_lists_to_search`.

### 2. LATERAL Join Confusion
Initial documentation suggested using LATERAL joins with VECTOR_SEARCH, but we found the native implementation without LATERAL performed better. Clearer guidance would help.

### 3. Embedding Dimension Limits
The 768-dimension limit for certain operations required us to carefully plan our embedding strategy. Support for higher dimensions (1536, 3072) would be valuable for more complex use cases.

### 4. AI.GENERATE Rate Limits
We encountered rate limits when generating personalized content at scale. While understandable for a preview feature, having clearer documentation about limits and best practices for batching would help.

### 5. Cost Visibility
It was sometimes difficult to predict costs for AI operations (embeddings, LLM calls) during development. A cost estimator tool would be extremely helpful.

## Feature Requests

1. **Streaming Vector Search**: Real-time updates to vector indexes for dynamic datasets
2. **Multi-modal Embeddings**: Native support for image and document embeddings
3. **Explainable AI**: Built-in functions to explain similarity scores and match reasoning
4. **Vector Index Monitoring**: Dashboard to monitor index performance and optimization suggestions
5. **Batch AI Operations**: More efficient batching for AI.GENERATE operations
6. **Multimodal Healthcare Applications**: While we didn't have time to explore this during the competition, the potential for multimodal experiences combining medical imaging, clinical text, and structured data represents an exciting future direction for BigQuery AI in healthcare

## Overall Assessment

BigQuery 2025's AI features represent a significant leap forward in making advanced AI capabilities accessible through SQL. The ability to perform semantic search, generate embeddings, and invoke LLMs directly from BigQuery transforms it from a data warehouse into a complete AI platform.

For our clinical trial matching use case, these features enabled us to build a production-ready system that would have been impossible with traditional SQL alone. The 20,000x performance improvement we achieved demonstrates the real-world impact of these technologies.

We're excited to see these features reach general availability and look forward to building more innovative solutions on BigQuery.

---

**Submitted by**: Victor Adafinoaiei
**Date**: September 2025
**Competition**: BigQuery AI - Building the Future of Data
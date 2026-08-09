                         ┌─────────────────┐
                         │   Streamlit UI  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Authentication  │
                         │ Authorization   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Agentic RAG    │
                         │    Planner      │
                         └────────┬────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
                ▼                 ▼                 ▼
          KB Search          Web Search       Metadata Search
                │                 │                 │
                ▼                 │                 │
       Hybrid Search              │                 │
       BM25 + Chroma              │                 │
                │                 │                 │
                ▼                 │                 │
       Cross Encoder              │                 │
         Reranker                 │                 │
                │                 │                 │
                └─────────────────┼─────────────────┘
                                  ▼
                         ┌─────────────────┐
                         │ Context        │
                         │ Evaluator      │
                         └────────┬────────┘
                                  │
                       ┌──────────┴──────────┐
                       │                     │
                  Insufficient          Sufficient
                       │                     │
                       ▼                     ▼
                 Reformulate              Groq
                    Query                   │
                       │                     ▼
                       └──────────────► Final Answer



----------------------

Question
↓
Planner
↓
Choose KB / WEB / BOTH
↓
Retrieve
↓
Rerank
↓
Evaluate
│
├── SUFFICIENT → Answer
│
└── INSUFFICIENT
↓
Reformulate query
↓
Search again
                     User
                      │
                      ▼
                 Streamlit UI
                      │
              Authentication
                      │
           ┌──────────┴──────────┐
           │                     │
    Security Context         Question
            │                     │
    owner/department/team         │
      visibility                  │
            │                     │
            └──────────┬──────────┘
                       ▼
                ask_agent()
                    │
                    ▼
                Groq Agent  
                      │
            ┌─────────┼─────────┐
            ▼         ▼         ▼
            KB        WEB      KB_WEB
            │                   │
            └─────────┬─────────┘
                      ▼
                query.py
                    │
                Metadata Filtering
                    │
                Hybrid Search
                    │
                Reranker
                    │
                  Groq
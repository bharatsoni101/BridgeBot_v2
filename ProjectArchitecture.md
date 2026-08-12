                         USER
                           │
                           ▼
                      Streamlit
                           │
                           ▼
                     ask_agent()
                           │
                           ▼
                    ┌─────────────┐
                    │    Agent    │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
             KB           WEB         KB_WEB
              │             │            │
              ▼             │            ▼
       Hybrid Search        │      Hybrid Search
              │             │            │
              ▼             │            ▼
         CrossEncoder       │       CrossEncoder
              │             │            │
              ▼             │            ▼
      Context Evaluator     │     Context Evaluator
              │             │            │
        ┌─────┴─────┐       │      ┌─────┴─────┐
        ▼           ▼       │      ▼           ▼
    SUFFICIENT  INSUFFICIENT│ SUFFICIENT  INSUFFICIENT
        │           │       │      │           │
        ▼           └───────┼──────┘           │
     Answer                 │                  │
                            ▼                  ▼
                            WEB              WEB
                            │                  │
                            └────────┬─────────┘
                                     ▼
                                Final Answer
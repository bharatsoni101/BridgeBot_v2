comand to start : API server 
===> uvicorn api.main:app --reload



Phase 3.5

                         USER
                           │
                           ▼
                      ask_agent()
                           │
                           ▼
                     ┌───────────┐
                     │   Agent   │
                     └─────┬─────┘
                           │
                  Initial Decision
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
             KB           WEB         KB_WEB
              │            │            │
              ▼            ▼            ▼
          Retrieval     Retrieval    Retrieval
              │            │            │
              └────────────┼────────────┘
                           ▼
                    Cross Encoder
                           │
                           ▼
                  Context Evaluator
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
             SUFFICIENT          INSUFFICIENT
                 │                   │
                 ▼                   ▼
             Answer          Agent chooses next
                                   action
                                      │
                           ┌──────────┼──────────┐
                           ▼          ▼          ▼
                          KB         WEB       KB_WEB
                           │          │          │
                           └──────────┼──────────┘
                                      ▼
                               Context Evaluator
                                      │
                               ┌──────┴──────┐
                               ▼             ▼
                           Sufficient    Insufficient
                               │             │
                               ▼             ▼
                            Answer       Safe Stop






--------------------------------------------------------------
Phase 3.0 
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
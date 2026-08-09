Authenticated User
        │
        ├── owner
        ├── department
        ├── team
        └── visibility
                │
                ▼
            ask_agent()
                │
                ▼
            Agent decides
            ┌─────┼─────┐
            ▼     ▼     ▼
            KB    WEB   KB_WEB
            │           │
            └─────┬─────┘
                  ▼
            query.ask()
                  │
                  ▼
        Metadata Filtering
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
        owner department  team
                │
                ▼
        Hybrid Search
                │
                ▼
        Cross Encoder
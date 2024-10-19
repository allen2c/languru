chat_openai_basic = {
    "summary": "OpenAI",
    "description": "Chat completion request",
    "value": {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    },
}
chat_openai_stream = {
    "summary": "OpenAI Stream",
    "description": "Chat completion stream request",
    "value": {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Respond accurately and concisely."},
            {"role": "user", "content": "Why is the sky blue?"},
        ],
        "stream": True,
    },
}
chat_google_gemini = {
    "summary": "Google Gemini",
    "description": "Chat completion request",
    "value": {
        "model": "gemini-1.5-flash",
        "messages": [
            {"role": "system", "content": "Respond accurately and concisely."},
            {"role": "user", "content": "Why is the banana yellow?"},
        ],
    },
}
chat_google_alias = {
    "summary": "Google Alias Name",
    "description": "Chat completion request",
    "value": {
        "model": "models/gemini-1.5-flash",
        "messages": [
            {"role": "system", "content": "Respond accurately and concisely."},
            {"role": "user", "content": "Why is the cherry red?"},
        ],
    },
}
chat_google_with_org = {
    "summary": "Google With Organization",
    "description": "Chat completion request",
    "value": {
        "model": "google/models/gemini-1.5-flash",
        "messages": [
            {"role": "system", "content": "Respond accurately and concisely."},
            {"role": "user", "content": "Why is the dragonfly blue?"},
        ],
    },
}
chat_perplexity = {
    "summary": "Perplexity Online",
    "description": "Chat completion request",
    "value": {
        "model": "llama-3.1-sonar-small-128k-chat",
        "messages": [
            {"role": "system", "content": "Respond accurately and concisely."},
            {"role": "user", "content": "Why is the apple red?"},
        ],
    },
}
chat_groq_mixtral = {
    "summary": "Groq Mixtral",
    "description": "Chat completion request",
    "value": {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are an unhelpful assistant"},
            {"role": "user", "content": "Are you a fish?"},
        ],
    },
}


chat_openapi_examples = {
    "OpenAI": chat_openai_basic,
    "OpenAI Stream": chat_openai_stream,
    "Google Alias": chat_google_alias,
    "Google Organization": chat_google_with_org,
    "Google Gemini": chat_google_gemini,
    "Perplexity Online": chat_perplexity,
    "Groq Mixtral": chat_groq_mixtral,
}

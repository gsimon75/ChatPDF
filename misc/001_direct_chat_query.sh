#!/bin/bash

read -r -d '' QUERY <<-END-OF-QUERY
{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": "What is Port D"
        }
    ]
}
END-OF-QUERY

curl https://api.openai.com/v1/chat/completions -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" -d "$QUERY"

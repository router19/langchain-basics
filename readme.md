#Using Open AI LLM
1. Create a .env file at root directory
2. Get the Open AI API secret key from Open AI API Platform 
3. Add a line in .env file with value 
```
OPENAI_API_KEY=<api-key>
```

For Using Langsmith for obserbaility 
set below variables in .env file, it would automatically be picked by langsmith website
Just be sure to use the same project name below and in langsmith
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<api-key>
LANGSMITH_PROJECT=langchaindemo
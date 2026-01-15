import ollama

class OllamaClient:
    def __init__(self, model="deepseek-r1:8b-0528-qwen3-q4_K_M"):
        self.model = model
        
    def generate(self, prompt):
        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            return response['response']
        except Exception as e:
            return f"Error communicating with Ollama: {e}\nEnsure Ollama is running and model '{self.model}' is pulled."

    def chat_loop(self, rag_engine):
        print("--- Smart Urban RAG Assistant (type 'quit' to exit) ---")
        while True:
            query = input("\nUser: ")
            if query.lower() in ['quit', 'exit']:
                break
                
            # 1. Retrieve
            print("Retrieving context...")
            results = rag_engine.retrieve(query, k=5)
            
            # 2. Format
            prompt = rag_engine.format_prompt(query, results)
            # print(f"DEBUG PROMPT:\n{prompt}\n")
            
            # 3. Generate
            print("Thinking...")
            answer = self.generate(prompt)
            print(f"\nAssistant: {answer}")

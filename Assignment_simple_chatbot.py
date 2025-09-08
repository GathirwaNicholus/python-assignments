print("Hello! I'm a simple AI & Data Science chatbot.")
print("Ask me something about Data Science or AI, or type 'bye' to exit.")

while True:
    user_input = input("You: ").lower()

    if user_input == "bye":
        print("Chatbot: Goodbye!")
        break
    elif "data science" in user_input:
        print("Chatbot: Data science is the study of data to extract meaningful insights.")
    elif "ai" in user_input or "artificial intelligence" in user_input:
        print("Chatbot: AI is the simulation of human intelligence by machines.")
    elif "machine learning" in user_input:
        print("Chatbot: Machine learning is a subset of AI that learns from data.")
    elif "deep learning" in user_input:
        print("Chatbot: Deep learning uses neural networks with many layers to model complex patterns.")
    else:
        print("Chatbot: I'm not sure about that. Try asking something else about AI or Data Science.")

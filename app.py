import gradio as gr
from gpt4all import GPT4All

model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")

def chatbot_response(input_text):
    tokens = []
    with model.chat_session() as session:
        for token in model.generate(input_text, streaming=True):
            tokens.append(token)
    response = ''.join(tokens)
    return response

iface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.components.Textbox(lines=2, placeholder="Ask me anything..."),
    outputs="text",
    title="GPT-4All Chatbot",
    description="This chatbot uses the GPT-4All model to answer your questions."
)

if __name__ == "__main__":
    iface.launch()

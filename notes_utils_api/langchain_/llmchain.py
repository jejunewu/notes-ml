from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(
    temperature=0,
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8888/v1",
    model_name="chatglm-6b-int4"
)

template = "你好请帮将文字从 {input_language} 翻译至 {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt)
content = chain.run(input_language="English", output_language="Chinese", text="I love programming.")
print(content)

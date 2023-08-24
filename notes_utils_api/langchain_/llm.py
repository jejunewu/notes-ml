from langchain import PromptTemplate, OpenAI, LLMChain

prompt_template = "请你帮我的{product}取个好听的名字"

llm = OpenAI(
    temperature=0,
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8888/v1",
    model_name="chatglm-6b-int4"
)
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)
content = llm_chain("衣服")
print(content)

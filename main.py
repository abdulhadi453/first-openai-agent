from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
from openai.types.responses import ResponseTextDeltaEvent
import os 
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


backend_agent = Agent(
    name="Backend Expert",
    instructions="""You are a backend expert. You help user with backend topics like APIs, Databases, authentication, server frameworks (e.g. Express.js, Django)
    Do not answer frontend or UI questions.
    """
)

frontend_agent = Agent(
    name="Front Expert",
    instructions="""You are a Frontend expert. You help with UI/UX using HTML, CSS, Javascript, React, Next.js and Tailwind CSS.
    
    Do not answer backend-related questions.
    """
)

web_dev_agent = Agent(
    name="Web Developer Agent",
    instructions="""You are a generalist web developer who decides wheather a question is about a frontend or backend.
    If the user ask about UI/UX realetd like HTML, CSS, Javascript, React, Next.js, tailwind etc, Handoff to the frontend developer.
    If the user ask about APIs, Databases, authentication, server frameworks (e.g. Express.js, Django) etc, Handoff to the backend developer.
    If it's unrelated to both politely decline.
    """,
    handoffs=[frontend_agent, backend_agent]
)

@cl.on_chat_start
async def handle_start():
    cl.user_session.set("history",[])
    await cl.Message(content="Hello from Hadi How may I assist you today").send()

@cl.on_message
async def handle_message(message : cl.Message):

    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})
    
    msg = cl.Message(content="")
    await msg.send()
    
    result = Runner.run_streamed(
        web_dev_agent,
        input=history,
        run_config=config
    )
    
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance (event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
    
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    # await cl.Message(content = result.final_output).send()
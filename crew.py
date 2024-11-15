from crewai import Crew,Process
from agents import blog_researcher,blog_writer
from tasks import research_task,write_task
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")
llm= ChatNVIDIA(model='nvidia/llama-3.1-nemotron-70b-instruct')
crew = Crew(
    agents=[blog_researcher, blog_writer],
    manager_llm = llm,
    tasks=[research_task, write_task],
    process=Process.sequential, 
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

result=crew.kickoff(inputs={'topic':'2025 Akrapovic Mercedes-AMG GLE 63 Coupe 820 - Wild SUV by German Classic Design'})
print(result)
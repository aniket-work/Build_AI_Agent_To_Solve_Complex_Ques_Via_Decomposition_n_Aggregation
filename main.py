import asyncio
import logging
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

from constants import LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT, EMBED_MODEL, EMBED_TIMEOUT, MAX_CONCURRENCY, WORKFLOW_TIMEOUT
from data_ingestion_1 import generate_pdfs
from query_engine import SubQuestionQueryEngine, create_query_engine_tools

# Set up detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    try:
        llm = Ollama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, timeout=LLM_TIMEOUT)
        Settings.llm = llm

        embed_model = OllamaEmbedding(model_name=EMBED_MODEL, timeout=EMBED_TIMEOUT)
        Settings.embed_model = embed_model

        # Generate PDFs if they are not already present
        await generate_pdfs()

        query_engine_tools = await create_query_engine_tools(llm, embed_model, semaphore)

        engine = SubQuestionQueryEngine(timeout=WORKFLOW_TIMEOUT, verbose=True)
        logger.info("Starting SubQuestionQueryEngine")
        result = await engine.run(
            llm=llm,
            tools=query_engine_tools,
            query="What has been the impact of Electric Vehicles on urban traffic from 2015 to 2023?"
        )

        logger.info(f"Final result: {result}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
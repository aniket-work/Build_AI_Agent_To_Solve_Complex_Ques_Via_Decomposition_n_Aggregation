import asyncio
import json
import logging
import re, os
from typing import List

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.workflow import step, Context, Workflow, Event, StartEvent, StopEvent
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from constants import REPORTS_FOLDER, STORAGE_FOLDER

logger = logging.getLogger(__name__)

class QueryEvent(Event):
    question: str

class AnswerEvent(Event):
    question: str
    answer: str

class SubQuestionQueryEngine(Workflow):

    @step(pass_context=True)
    async def query(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        if hasattr(ev, "query"):
            ctx.data["original_query"] = ev.query
            logger.info(f"Query is {ctx.data['original_query']}")

        if hasattr(ev, "llm"):
            ctx.data["llm"] = ev.llm

        if hasattr(ev, "tools"):
            ctx.data["tools"] = ev.tools

        prompt = f"""
            Given a user question and a list of tools, output a list of
            relevant sub-questions. The answers to all sub-questions should,
            when combined, answer the original question. Respond in JSON format:
            {{
                "sub_questions": [
                    "How has EV adoption grown from 2015 to 2023?",
                    "What impact have EVs had on traffic congestion in major cities?",
                    "What are the future trends in EV adoption?"
                ]
            }}
            User question: {ctx.data['original_query']}
            Available tools: {[tool.metadata.name for tool in ctx.data['tools']]}
        """

        try:
            response = await self.ollama_complete_with_retry(ctx.data["llm"], prompt)
            logger.info(f"Raw response from Ollama: {response}")

            clean_response = re.sub(r'```json\s*|\s*```', '', response).strip()

            try:
                response_obj = json.loads(clean_response)
                sub_questions = response_obj.get("sub_questions", [])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {str(e)}")
                logger.error(f"Cleaned response: {clean_response}")
                return QueryEvent(question="Error occurred while generating sub-questions")

            ctx.data["sub_question_count"] = len(sub_questions)

            for question in sub_questions:
                ctx.send_event(QueryEvent(question=question))

        except Exception as e:
            logger.error(f"Error in query step: {str(e)}")
            return QueryEvent(question="Error occurred while generating sub-questions")

        return QueryEvent(question="Sub-questions generated")

    @step(pass_context=True)
    async def sub_question(self, ctx: Context, ev: QueryEvent) -> AnswerEvent:
        logger.info(f"Processing sub-question: {ev.question}")
        agent = ReActAgent.from_tools(ctx.data["tools"], llm=ctx.data["llm"], verbose=True)

        try:
            response = await asyncio.wait_for(agent.chat(ev.question), timeout=1800)  # 30 minutes timeout
            logger.info(f"Response for sub-question '{ev.question}': {response}")
            if not response.get('text'):
                raise ValueError("Empty response received")
            return AnswerEvent(question=ev.question, answer=str(response['text']))
        except asyncio.TimeoutError:
            logger.warning(f"Timeout occurred while processing sub-question: {ev.question}")
            return AnswerEvent(question=ev.question, answer="Timeout occurred while processing this sub-question")
        except Exception as e:
            logger.error(f"Error processing sub-question: {ev.question}. Error: {str(e)}")
            return AnswerEvent(question=ev.question, answer=f"Error: {str(e)}")

    @step(pass_context=True)
    async def combine_answers(self, ctx: Context, ev: AnswerEvent) -> StopEvent:
        ready = ctx.collect_events(ev, [AnswerEvent] * ctx.data.get("sub_question_count", 0))
        if not ready:
            logger.warning("No answers were received for sub-questions.")
            return StopEvent(result="Failed to generate answers to sub-questions.")

        answers = "\n\n".join([f"Question: {event.question}\nAnswer: {event.answer}" for event in ready])

        prompt = f"""
            Combine the answers to these sub-questions into a single answer to the original question.

            Original question: {ctx.data['original_query']}

            Sub-questions and answers:
            {answers}
        """

        logger.info(f"Final prompt is:\n{prompt}")

        try:
            response = await self.ollama_complete_with_retry(ctx.data["llm"], prompt)
            logger.info("Final response: %s", response)
        except Exception as e:
            logger.error(f"Error combining answers: {str(e)}")
            response = f"Error: {str(e)}"

        return StopEvent(result=str(response))

    async def ollama_complete_with_retry(self, llm: Ollama, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(llm.acomplete(prompt), timeout=1200)  # 20 minutes timeout
                return str(response)
            except asyncio.TimeoutError:
                logger.warning(f"Ollama request timed out. Attempt {attempt + 1} of {max_retries}")
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                logger.error(f"Error in Ollama request: {str(e)}")
                if attempt == max_retries - 1:
                    raise
            await asyncio.sleep(5 * (attempt + 1))  # Exponential backoff

async def process_file(file: str, folder: str, llm: Ollama, embed_model: OllamaEmbedding, semaphore: asyncio.Semaphore) -> QueryEngineTool:
    year = file.split("_")[0]
    index_persist_path = f"./storage/ev_reports_{year}/"

    async with semaphore:
        try:
            logger.info(f"Processing file: {file}")
            if os.path.exists(index_persist_path):
                logger.info(f"Loading existing index for {year}")
                storage_context = StorageContext.from_defaults(persist_dir=index_persist_path)
                index = load_index_from_storage(storage_context, embed_model=embed_model)
            else:
                logger.info(f"Creating new index for {year}")
                documents = SimpleDirectoryReader(input_files=[folder + file]).load_data()
                index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
                index.storage_context.persist(index_persist_path)

            logger.info(f"Creating query engine for {year}")
            engine = index.as_query_engine(llm=llm)
            return QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(
                    name=f"ev_reports_{year}",
                    description=f"Analysis of Electric Vehicle (EV) impact on urban traffic in {year}",
                ),
            )
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}", exc_info=True)
            return None

async def create_query_engine_tools(llm: Ollama, embed_model: OllamaEmbedding, semaphore: asyncio.Semaphore) -> List[QueryEngineTool]:
    files = [f for f in os.listdir(REPORTS_FOLDER) if f.endswith('.pdf')]
    query_engine_tools = []
    batch_size = 1
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        logger.debug(f"Processing batch {i//batch_size + 1}")
        tasks = [process_file(file, REPORTS_FOLDER, llm, embed_model, semaphore) for file in batch]
        batch_results = await asyncio.gather(*tasks)
        query_engine_tools.extend([tool for tool in batch_results if tool is not None])
    return query_engine_tools
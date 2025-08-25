import asyncio
import os
from typing import List

import loguru
from llama_index.core import SQLDatabase, VectorStoreIndex
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.indices.struct_store.sql_retriever import SQLRetriever
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema, ObjectIndex
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent, Context, Event
from llama_index.llms.deepseek import DeepSeek
from sqlalchemy import create_engine

from data_prepare import gen_table_info_and_store, get_csv_infos

logger = loguru.logger
def build_obj_retriever(sql_database, table_infos):
    """
    构建对象索引
    :param sql_database:
    :param table_infos:
    :return:
    """

    table_node_mapping = SQLTableNodeMapping(
        sql_database=sql_database,
    )
    table_schema_object = [
        SQLTableSchema(table_name=t.table_name,
                       context_str=f"table_summary: {t.table_summary} table_columns: {t.columns}")  for t in table_infos
    ]
    obj_index = ObjectIndex.from_objects(
        objects=table_schema_object,
        object_mapping=table_node_mapping,
        index_cls=VectorStoreIndex,
    )

    obj_retriever = obj_index.as_retriever(similarity_top_k=3)
    return obj_retriever

class TableRetrieveEvent(Event):
    table_context_str:str
    query:str

class TextToSQLEvent(Event):
    sql:str
    query:str

class TextToSQLWorkFlow(Workflow):
    def __init__(
        self,
        table_infos,
        obj_retriever,
        sql_retriever,
        text2sql_prompt,
        response_synthesizer_prompt,
        llm,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.table_infos = table_infos
        self.obj_retriever = obj_retriever
        self.sql_retriever = sql_retriever
        self.text2sql_prompt = text2sql_prompt
        self.response_synthesizer_prompt = response_synthesizer_prompt
        self.llm:DeepSeek = llm

    def get_table_context_str(self, table_schema_objs:List[SQLTableSchema]) -> str:
        table_contexts = []
        for table_schema_obj in table_schema_objs:
            table_context = f"表 {table_schema_obj.table_name} 的描述信息: {table_schema_obj.context_str}.\n"
            table_contexts.append(table_context)
        return "\n".join(table_contexts)

    @step
    def retriever_table(self, ctx:Context, ev:StartEvent) -> TableRetrieveEvent:
        """检索表"""
        table_schema_objs=self.obj_retriever.retrieve(ev.query)
        table_context_str = self.get_table_context_str(table_schema_objs)
        return TableRetrieveEvent(table_context_str=table_context_str, query=ev.query)

    def parse_response_to_sql(self, chat_resp:ChatResponse)->str:
        """解析LLM返回的SQL"""
        response = chat_resp.message.content
        sql_query_start = response.find("SQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            if response.startswith("SQLQuery:"):
                response = response[len("SQLQuery:"):].strip()
        sql_result_start = response.find("SQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start].strip()
        return response.strip().strip("```").strip()

    @step
    def gen_sql(self, ctx:Context, ev:TableRetrieveEvent)->TextToSQLEvent:
        """生成SQL"""
        prompt = self.text2sql_prompt.format_messages(
            schema=ev.table_context_str,
            query=ev.query,
        )

        chat_response = self.llm.chat(prompt)
        sql = self.parse_response_to_sql(chat_response)
        logger.info(f"sql: {sql}")
        return TextToSQLEvent(sql=sql, query=ev.query)

    @step
    def gen_response(self, ctx:Context, ev:TextToSQLEvent)->StopEvent:
        """生成回答"""
        sql = ev.sql
        query = ev.query
        sql_response = self.sql_retriever.retrieve(sql)
        prompt = self.response_synthesizer_prompt.format(
            query=query,
            sql=sql,
            sql_response=sql_response,
        )
        prompt = [ChatMessage.from_str(prompt, role="user")]
        response = self.llm.chat(prompt)
        logger.info(f"user_input: {query}")
        logger.info(f"response: {response}")
        return StopEvent(response=response)



async def main():
    csv_dir = './data/'
    local_table_info_dir = "./data/table_info"
    if os.path.exists(csv_dir):
        csv_dir = "./data/WikiTableQuestions/csv/200-csv"

    db_path="sqlite:///./data/wiki_table_questions.db"
    engine = create_engine(db_path)
    sql_database = SQLDatabase(engine=engine)
    csv_dfs = get_csv_infos(csv_dir)
    # 1. 获取table_infos
    table_infos = gen_table_info_and_store(csv_dfs, local_table_info_dir)
    # 2. 构建object_retriever
    object_retriever = build_obj_retriever(sql_database, table_infos)
    # 3. 构建sql_retriever
    sql_retriever = SQLRetriever(sql_database)
    # 4. 构建workflow
    response_synthesizer_prompt = """
    Given an input question, SQL query, and SQL query result, generate a final answer.
    Query: {query}
    SQL Query: {sql}
    SQL Result: {sql_response}
    Final Answer:
    """
    flow = TextToSQLWorkFlow(
        table_infos,
        obj_retriever=object_retriever,
        sql_retriever=sql_retriever,
        text2sql_prompt=DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(dialect=engine.dialect.name), # 默认的模板
        response_synthesizer_prompt=response_synthesizer_prompt,
        llm=DeepSeek(model="deepseek-reasoner", api_key=os.environ["DEEPSEEK_API_KEY"]),
        timeout=None
    )
    # 5. 执行workflow
    query = "What was the year that Diddy was signed to Bad Boy?"
    response = await flow.run(query=query)
    return response

if __name__ == '__main__':
    asyncio.run(main())
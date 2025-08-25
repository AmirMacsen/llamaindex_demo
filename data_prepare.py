import json
import os
import pathlib
from typing import List

import loguru
from pandas import DataFrame
from pydantic import BaseModel, Field
from sqlalchemy import Column

import setup_llamaindex

setup_llamaindex.setup_llama_index()

logger = loguru.logger


def download_zip_and_extract(url, extract_to='.'):
    import os
    import zipfile
    import requests
    from tqdm import tqdm

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    logger.info(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024
    output_path = os.path.join(extract_to, "download.zip")
    with open(output_path, 'wb') as file, tqdm(
            desc=output_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            file.write(data)
            bar.update(len(data))

    logger.info(f"Finished downloading from {url}...")
    with zipfile.ZipFile(os.path.join(extract_to, 'download.zip'), 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    logger.info(f"Extracted to {extract_to}")


class TableInfo(BaseModel):
    table_name: str = Field(..., description="The name of the table")
    table_summary: str = Field(..., description="A brief summary of the table")
    columns:list = Field(..., description="Sanitized column names of the table")


def get_csv_infos(file_path: str) -> List[DataFrame]:
    """
    获取csv数据
    :param file_path:
    :return:
    """
    import pandas as pd
    # 读取csv文件
    dfs = []
    logger.info(f"Reading dir {file_path}...")
    csv_files = sorted(pathlib.Path(file_path).rglob("*.csv"))
    for file in csv_files:
        try:
            logger.info(f"Reading {file}...")
            df = pd.read_csv(file).dropna()
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to read {file}: {e}")
    return dfs


def _get_table_info_with_idx(file_path, idx):
    """
    从本地获取表的信息，如果没有则返回None，如果有多个则报错
    表的信息存储在file_path目录下，文件名格式为{idx}_*.json
    """
    table_info_list = list(pathlib.Path(file_path).rglob(f"{idx}_*"))
    if len(table_info_list) == 0:
        return None
    elif len(table_info_list) == 1:
        path = table_info_list[0]
        with open(path, "r", encoding="utf-8") as f:
            table_info = TableInfo(**json.load(f))
        return table_info
    else:
        raise ValueError(f"Multiple table info files found for index {idx} in {file_path}")


def gen_table_info_and_store(dfs: List[DataFrame], store_dir: str) -> List[TableInfo]:
    """
    通过大模型生成表的信息，包含表名称和表的描述信息
    :return: List[TableInfo]
    """
    import os
    from llama_index.core import ChatPromptTemplate
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.llms.deepseek import DeepSeek

    prompt_str = """
    请用以下 JSON 格式给我一个表格的摘要。
    - 表名必须是表的唯一名称，表名必须简洁并且唯一。
    - 不要输出通用表名（例如表、my_table）。
    - 表名称用小写字母，除此之外你不能更改。
    - 字段名称不要修改。
    - 表的摘要描述中，你需要根据传入的数据和字段名称，生成字段的描述信息以及其他必要信息。
    - 禁止无关信息。
    不要将表名设为以下之一：{exclude_table_name_list}

    Table:
    {table_str}
    Columns:
    {columns_str}
    Summary:"""
    prompt_tmpl = ChatPromptTemplate(
        message_templates=[ChatMessage.from_str(prompt_str, role="user")],
    )
    llm = DeepSeek(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )
    table_infos = []
    table_names = []
    for idx, df in enumerate(dfs):
        table_info = _get_table_info_with_idx(store_dir, idx)
        if table_info:
            table_infos.append(table_info)
            table_names.append(table_info.table_name)
            continue
        df_str = df.head(10).to_csv()
        sanitized_columns = [sanitize_column_name(col) for col in df.columns]
        columns_str = ", ".join(sanitized_columns)
        print(columns_str)
        table_info = llm.structured_predict(
            output_cls=TableInfo,
            prompt=prompt_tmpl,
            table_str=df_str,
            exclude_table_name_list=str(list(table_names)),
            columns_str=columns_str
        )
        table_name = table_info.table_name
        logger.info(f"Processing table: {table_name}")
        if table_name not in table_names:
            table_names.append(table_name)
            if not os.path.exists(store_dir):
                os.makedirs(store_dir)
            with open(os.path.join(store_dir, f"{idx}_{table_name}.json"), "w", encoding="utf-8") as f:
                json.dump(table_info.dict(), f, indent=2)
            table_infos.append(table_info)
        else:
            logger.warning(f"Duplicate table name found: {table_name}, skipping...")
    return table_infos


def sanitize_column_name(col_name):
    """
    调整列名，去除特殊字符，替换为空格
    :param col_name:
    :return:
    """
    import re
    return re.sub(r"\W+", "_", col_name)


def store_csv_data_to_sqlit(dfs: List[DataFrame], file_path: str, db_path: str):
    """
    将表数据存储到sqlite数据库中
    :param dfs:
    :param db_path:
    :param file_path:
    :return:
    """
    from sqlalchemy import String, Integer, Table, create_engine, MetaData
    engine = create_engine(f"sqlite:///{db_path}")
    meta_obj = MetaData()
    for idx, df in enumerate(dfs):
        table_info = _get_table_info_with_idx(file_path, idx)
        sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
        df = df.rename(columns=sanitized_columns)
        columns = [
            Column(col, String if dtype == 'object' else Integer) for col, dtype in
            zip(df.columns, df.dtypes.astype(str))
        ]

        table = Table(table_info.table_name.lower(), meta_obj, *columns)
        meta_obj.create_all(engine)

        with engine.connect() as conn:
            for _, row in df.iterrows():
                insert_stmt = table.insert().values(**row.to_dict())
                try:
                    conn.execute(insert_stmt)
                except Exception as e:
                    logger.error(f"Failed to insert row into {table_info.table_name}: {e}")
            conn.commit()


if __name__ == '__main__':
    zip_url = "https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip"
    csv_dir = './data/'
    local_table_info_dir = "./data/table_info"
    db_path = "./data/wiki_table_questions.db"
    # download_zip_and_extract(zip_url, extract_to='./data')
    if os.path.exists(csv_dir):
        csv_dir = "./data/WikiTableQuestions/csv/200-csv"
    csv_dfs = get_csv_infos(csv_dir)
    table_infos = gen_table_info_and_store(csv_dfs, local_table_info_dir)
    for info in table_infos:
        logger.info(f"Table Name: {info.table_name}, Summary: {info.table_summary}")

    store_csv_data_to_sqlit(csv_dfs, local_table_info_dir, db_path)


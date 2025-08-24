"""
llamaindex setup script
for llms and embedding models
"""
import os
from typing import Optional, Any, List

import requests
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.deepseek import DeepSeek


def setup_llama_index():
    load_dotenv()

    Settings.llm = DeepSeek(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )


    class ZhipuAIEmbedding(BaseEmbedding):
        """
        自定义智谱清言 Embedding 模型
        请根据智谱清言官方 API 文档调整参数和请求格式
        """

        def __init__(
                self,
                model_name: str = "embedding-3",  # 替换为智谱清言具体的嵌入模型名
                api_key: Optional[str] = None,
                api_base: str = "https://open.bigmodel.cn/api/paas/v4/",  # 假设的 API 端点，请根据实际情况修改
                **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self._model_name = model_name
            self._api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
            self._api_base = api_base.rstrip('/')  # 确保没有末尾的斜杠

            if not self._api_key:
                raise ValueError(
                    "ZhipuAI API key is required. Either pass it as an argument "
                    "or set the ZHIPUAI_API_KEY environment variable."
                )

        def _get_query_embedding(self, query: str) -> List[float]:
            # 查询和文档通常使用相同的模型处理
            return self._get_text_embedding(query)

        def _get_text_embedding(self, text: str) -> List[float]:
            # 构建请求 API 的 URL
            url = f"{self._api_base}/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}"
            }
            # 根据智谱清言的 API 要求构建数据体
            data = {
                "model": self._model_name,
                "input": text
                # 可能还需要其他参数，如 "encoding_format" 等，请查阅官方文档
            }

            response = requests.post(url, headers=headers, json=data)

            # 检查请求是否成功
            if response.status_code != 200:
                raise Exception(f"ZhipuAI API request failed with status {response.status_code}: {response.text}")

            # 解析响应，这里假设返回的 JSON 结构中有 `data` 列表，且第一个元素包含 `embedding`
            result = response.json()
            # 以下路径解析需要根据智谱清言 API 的实际返回格式进行调整
            if 'data' in result and len(result['data']) > 0:
                return result['data'][0]['embedding']
            else:
                raise Exception(f"Unexpected response format from ZhipuAI API: {result}")

        def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
            # 批量处理文本嵌入
            url = f"{self._api_base}/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}"
            }
            # 根据智谱清言的 API 要求构建数据体，支持批量输入
            data = {
                "model": self._model_name,
                "input": texts
                # 可能还需要其他参数
            }

            response = requests.post(url, headers=headers, json=data)

            if response.status_code != 200:
                raise Exception(f"ZhipuAI API request failed with status {response.status_code}: {response.text}")

            result = response.json()
            # 以下路径解析需要根据智谱清言 API 的实际返回格式进行调整
            if 'data' in result:
                return [item['embedding'] for item in result['data']]
            else:
                raise Exception(f"Unexpected response format from ZhipuAI API: {result}")

        async def _aget_query_embedding(self, query: str) -> List[float]:
            return self._get_query_embedding(query)

        async def _aget_text_embedding(self, text: str) -> List[float]:
            return self._get_text_embedding(text)

        async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
            return self._get_text_embeddings(texts)

        @classmethod
        def class_name(cls) -> str:
            return "ZhipuAIEmbedding"

    Settings.embed_model = ZhipuAIEmbedding(
        model_name="embedding-3",
        api_key=os.environ["ZHIPUAI_API_KEY"],

    )



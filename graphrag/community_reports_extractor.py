# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
Reference:
 - [graphrag](https://github.com/microsoft/graphrag)
"""

import logging
import json
import re
import traceback
from typing import Callable
from dataclasses import dataclass
import networkx as nx
import pandas as pd
from graphrag import leiden
from graphrag.community_report_prompt import COMMUNITY_REPORT_PROMPT
from graphrag.extractor import Extractor
from graphrag.leiden import add_community_info2graph
from rag.llm.chat_model import Base as CompletionLLM
from graphrag.utils import ErrorHandlerFn, perform_variable_replacements, dict_has_keys_with_types
from rag.utils import num_tokens_from_string
from timeit import default_timer as timer


@dataclass
class CommunityReportsResult:
    """Community reports result class definition."""

    output: list[str]
    structured_output: list[dict]


class CommunityReportsExtractor(Extractor):
    """Community reports extractor class definition."""

    _extraction_prompt: str
    _output_formatter_prompt: str
    _on_error: ErrorHandlerFn
    _max_report_length: int

    def __init__(
        self,
        llm_invoker: CompletionLLM,
        extraction_prompt: str | None = None,
        on_error: ErrorHandlerFn | None = None,
        max_report_length: int | None = None,
    ):
        """Init method definition."""
        self._llm = llm_invoker
        self._extraction_prompt = extraction_prompt or COMMUNITY_REPORT_PROMPT
        self._on_error = on_error or (lambda _e, _s, _d: None)
        self._max_report_length = max_report_length or 1500

    def __call__(self, graph: nx.Graph, callback: Callable | None = None):
        ## leiden算法，每个层级的社区结构
        communities: dict[str, dict[str, list]] = leiden.run(graph, {})
        total = sum([len(comm.items()) for _, comm in communities.items()])
        relations_df = pd.DataFrame([{"source":s, "target": t, **attr} for s, t, attr in graph.edges(data=True)])
        res_str = []
        res_dict = []
        over, token_count = 0, 0
        st = timer()
        for level, comm in communities.items():
            for cm_id, ents in comm.items():
                weight = ents["weight"]
                ents = ents["nodes"]
                ## 实体数据
                ent_df = pd.DataFrame([{"entity": n, **graph.nodes[n]} for n in ents])
                ## 关系数据
                rela_df = relations_df[(relations_df["source"].isin(ents)) | (relations_df["target"].isin(ents))].reset_index(drop=True)
                ## 生成模板变量
                prompt_variables = {
                    "entity_df": ent_df.to_csv(index_label="id"),
                    "relation_df": rela_df.to_csv(index_label="id")
                }
                text = perform_variable_replacements(self._extraction_prompt, variables=prompt_variables)
                gen_conf = {"temperature": 0.3}
                try:
                    ## 大模型生成社区描述
                    response = self._chat(text, [{"role": "user", "content": "Output:"}], gen_conf)
                    token_count += num_tokens_from_string(text + response)
                    ## 输出格式化
                    response = re.sub(r"^[^\{]*", "", response)
                    response = re.sub(r"[^\}]*$", "", response)
                    response = re.sub(r"\{\{", "{", response)
                    response = re.sub(r"\}\}", "}", response)
                    logging.debug(response)
                    response = json.loads(response)
                    if not dict_has_keys_with_types(response, [
                                ("title", str),
                                ("summary", str),
                                ("findings", list),
                                ("rating", float),
                                ("rating_explanation", str),
                            ]):
                        continue
                    response["weight"] = weight
                    response["entities"] = ents
                except Exception as e:
                    logging.exception("CommunityReportsExtractor got exception")
                    self._on_error(e, traceback.format_exc(), None)
                    continue
                ## 将社区标题添加到节点上
                add_community_info2graph(graph, ents, response["title"])
                ## 保存输出和格式化输出
                res_str.append(self._get_text_output(response))
                res_dict.append(response)
                over += 1
                if callback:
                    callback(msg=f"Communities: {over}/{total}, elapsed: {timer() - st}s, used tokens: {token_count}")

        return CommunityReportsResult(
            structured_output=res_dict,
            output=res_str,
        )

    def _get_text_output(self, parsed_output: dict) -> str:
        title = parsed_output.get("title", "Report")
        summary = parsed_output.get("summary", "")
        findings = parsed_output.get("findings", [])

        def finding_summary(finding: dict):
            if isinstance(finding, str):
                return finding
            return finding.get("summary")

        def finding_explanation(finding: dict):
            if isinstance(finding, str):
                return ""
            return finding.get("explanation")

        report_sections = "\n\n".join(
            f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
        )
        return f"# {title}\n\n{summary}\n\n{report_sections}"

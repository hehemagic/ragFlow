#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import json
import math
import re
import logging
import copy
from elasticsearch_dsl import Q

from rag.nlp import rag_tokenizer, term_weight, synonym

class EsQueryer:
    def __init__(self, es):
        self.tw = term_weight.Dealer()
        self.es = es
        ## 同义词
        self.syn = synonym.Dealer()
        self.flds = ["ask_tks^10", "ask_small_tks"]

    @staticmethod
    def subSpecialChar(line):
        return re.sub(r"([:\{\}/\[\]\-\*\"\(\)\|\+~\^])", r"\\\1", line).strip()

    @staticmethod
    def isChinese(line):
        arr = re.split(r"[ \t]+", line)
        if len(arr) <= 3:
            return True
        e = 0
        for t in arr:
            if not re.match(r"[a-zA-Z]+$", t):
                e += 1
        return e * 1. / len(arr) >= 0.7

    @staticmethod
    def rmWWW(txt):
        """
        移除文本中的WWW（World Wide Web）相关词汇和英文代词、助动词等，以统一文本格式。
    
        参数:
        txt (str): 需要处理的文本字符串。
    
        返回值:
        str: 处理后的文本字符串。
        """
        # 定义需要移除或替换的文本模式列表
        patts = [
            # 移除中文询问词汇和语气词
            (r"是*(什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀)是*", ""),
            # 移除或替换英文疑问词
            (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
            # 移除或替换英文常见的代词、助动词和介词
            (r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down) ", " ")
        ]
        # 遍历模式列表，使用正则表达式替换文本中的匹配项
        for r, p in patts:
            txt = re.sub(r, p, txt, flags=re.IGNORECASE)
        return txt

    ## 构建查询条件
    def question(self, txt, tbl="qa", min_match="60%"):
        ## 处理原始文本
        txt = re.sub(
            r"[ :\r\n\t,，。？?/`!！&\^%%]+",
            " ",
            rag_tokenizer.tradi2simp(
                rag_tokenizer.strQ2B(
                    txt.lower()))).strip()
        txt = EsQueryer.rmWWW(txt)

        if not self.isChinese(txt):
            ## 不是中文
            tks = rag_tokenizer.tokenize(txt).split(" ")
            ## 计算权重
            tks_w = self.tw.weights(tks)
            tks_w = [(re.sub(r"[ \\\"'^]", "", tk), w) for tk, w in tks_w]
            tks_w = [(re.sub(r"^[a-z0-9]$", "", tk), w) for tk, w in tks_w if tk]
            tks_w = [(re.sub(r"^[\+-]", "", tk), w) for tk, w in tks_w if tk]
            q = ["{}^{:.4f}".format(tk, w) for tk, w in tks_w if tk]
            for i in range(1, len(tks_w)):
                ## 对于相邻词语，生成带有双词匹配的查询格式，权重是两个词语权重的最大值乘以 2
                q.append("\"%s %s\"^%.4f" % (tks_w[i - 1][0], tks_w[i][0], max(tks_w[i - 1][1], tks_w[i][1])*2))
            if not q:
                q.append(txt)
            return Q("bool",
                     must=Q("query_string", fields=self.flds,
                            type="best_fields", query=" ".join(q),
                            boost=1)#, minimum_should_match=min_match)
                     ), tks

        def need_fine_grained_tokenize(tk):
            if len(tk) < 3:
                return False
            if re.match(r"[0-9a-z\.\+#_\*-]+$", tk):
                return False
            return True

        qs, keywords = [], []
        for tt in self.tw.split(txt)[:256]:  # .split(" "):
            if not tt:
                continue
            twts = self.tw.weights([tt])
            ## 同义词处理
            syns = self.syn.lookup(tt)
            logging.info(json.dumps(twts, ensure_ascii=False))
            tms = []
            for tk, w in sorted(twts, key=lambda x: x[1] * -1):
                ## 按权重排序
                sm = rag_tokenizer.fine_grained_tokenize(tk).split(" ") if need_fine_grained_tokenize(tk) else []
                ## 去除无意义的符号
                sm = [
                    re.sub(
                        r"[ ,\./;'\[\]\\`~!@#$%\^&\*\(\)=\+_<>\?:\"\{\}\|，。；‘’【】、！￥……（）——《》？：“”-]+",
                        "",
                        m) for m in sm]
                sm = [EsQueryer.subSpecialChar(m) for m in sm if len(m) > 1]
                sm = [m for m in sm if len(m) > 1]
                ## 构建关键词列表
                keywords.append(re.sub(r"[ \\\"']+", "", tk))
                keywords.extend(sm)
                if len(keywords) >= 12: break

                ## 查询同义词，将同义词和原词组合
                tk_syns = self.syn.lookup(tk)
                tk = EsQueryer.subSpecialChar(tk)
                if tk.find(" ") > 0:
                    tk = "\"%s\"" % tk
                if tk_syns:
                    tk = f"({tk} %s)" % " ".join(tk_syns)
                if sm:
                    ## 存在细粒度分词, 将其加入查询表达式中，形成 tk、sm 和模糊匹配（编辑距离为 2）的查询组合，并且设置权重 ^0.5
                    tk = f"{tk} OR \"%s\" OR (\"%s\"~2)^0.5" % (
                        " ".join(sm), " ".join(sm))
                if tk.strip():
                    tms.append((tk, w))

            tms = " ".join([f"({t})^{w}" for t, w in tms])

            if len(twts) > 1:
                tms += f" (\"%s\"~4)^1.5" % (" ".join([t for t, _ in twts]))
            if re.match(r"[0-9a-z ]+$", tt):
                tms = f"(\"{tt}\" OR \"%s\")" % rag_tokenizer.tokenize(tt)

            syns = " OR ".join(
                ["\"%s\"^0.7" % EsQueryer.subSpecialChar(rag_tokenizer.tokenize(s)) for s in syns])
            if syns:
                tms = f"({tms})^5 OR ({syns})^0.7"

            qs.append(tms)

        flds = copy.deepcopy(self.flds)
        mst = []
        if qs:
            mst.append(
                Q("query_string", fields=flds, type="best_fields",
                  query=" OR ".join([f"({t})" for t in qs if t]), boost=1, minimum_should_match=min_match)
            )

        return Q("bool",
                 must=mst,
                 ), keywords

    def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3,
                          vtweight=0.7):
        from sklearn.metrics.pairwise import cosine_similarity as CosineSimilarity
        import numpy as np
        ## 向量相似度
        sims = CosineSimilarity([avec], bvecs)
        ## 词汇相似度
        tksim = self.token_similarity(atks, btkss)
        return np.array(sims[0]) * vtweight + \
            np.array(tksim) * tkweight, tksim, sims[0]

    def token_similarity(self, atks, btkss):
        def toDict(tks):
            d = {}
            if isinstance(tks, str):
                tks = tks.split(" ")
            for t, c in self.tw.weights(tks):
                if t not in d:
                    d[t] = 0
                d[t] += c
            return d
        ## key为词汇，value为词汇的权重
        atks = toDict(atks)
        btkss = [toDict(tks) for tks in btkss]
        return [self.similarity(atks, btks) for btks in btkss]

    ## 计算词汇相似度
    def similarity(self, qtwt, dtwt):
        if isinstance(dtwt, type("")):
            dtwt = {t: w for t, w in self.tw.weights(self.tw.split(dtwt))}
        if isinstance(qtwt, type("")):
            qtwt = {t: w for t, w in self.tw.weights(self.tw.split(qtwt))}
        s = 1e-9
        for k, v in qtwt.items():
            if k in dtwt:
                s += v  # * dtwt[k]
        q = 1e-9
        for k, v in qtwt.items():
            q += v  # * v
        #d = 1e-9
        # for k, v in dtwt.items():
        #    d += v * v
        return s / q / max(1, math.sqrt(math.log10(max(len(qtwt.keys()), len(dtwt.keys())))))# math.sqrt(q) / math.sqrt(d)

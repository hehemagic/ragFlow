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

from rag.nlp import find_codec,num_tokens_from_string
import re

class RAGFlowTxtParser:
    def __call__(self, fnm, binary=None, chunk_token_num=128, delimiter="\n!?;。；！？"):
        txt = ""
        if binary:
            ## 对应编码格式
            encoding = find_codec(binary)
            txt = binary.decode(encoding, errors="ignore")
        else:
            with open(fnm, "r") as f:
                while True:
                    l = f.readline()
                    if not l:
                        break
                    txt += l
        return self.parser_txt(txt, chunk_token_num, delimiter)

    @classmethod
    ## 按照分隔符简单切分
    def parser_txt(cls, txt, chunk_token_num=128, delimiter="\n!?;。；！？"):
        if type(txt) != str:
            raise TypeError("txt type should be str!")
        cks = [""]
        tk_nums = [0]

        def add_chunk(t):
            nonlocal cks, tk_nums, delimiter
            tnum = num_tokens_from_string(t)
            if tnum < 8:
                pos = ""
            if tk_nums[-1] > chunk_token_num:
                cks.append(t)
                tk_nums.append(tnum)
            else:
                cks[-1] += t
                tk_nums[-1] += tnum

        s, e = 0, 1
        while e < len(txt):
            if txt[e] in delimiter:
                add_chunk(txt[s: e + 1])
                s = e + 1
                e = s + 1
            else:
                e += 1
        if s < e:
            add_chunk(txt[s: e + 1])

        return [[c,""] for c in cks]
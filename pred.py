"""
The pred.py is used to recommendate department and similarity question.
If running, see http://127.0.0.1:8050/ on a brower

Runs in Python 3.12.4.

@author: Jin-Zhou Lee
"""

#%% packages
import numpy as np
import pandas as pd
import tomotopy as tp
from tomotopy.utils import Corpus
import fasttext

#%% word segmentate
import jieba
import unicodedata
import re

jieba.set_dictionary('model/dict.txt.big')

def pre_process(text):
    move_chr = r'[^\u4e00-\u9fffA-Za-z,\s]'
    
    # full-width font to half-width
    text = unicodedata.normalize("NFKD", text)
    
    # to lower
    text = str(text).lower()
    
    # Remove URL
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    
    # Remove non-Chinese characters and English
    text = re.sub(move_chr, '', text)
    
    # segmentate
    text = list(jieba.cut_for_search(text))
    
    # slove spaces problem
    text = ' '.join(text)
    
    # Merge multiple consecutive spaces and line breaks into a single space
    text = re.sub(r'[\s+,]+', ' ', text)
    
    # Remove leading and trailing spaces
    text = text.strip()
    
    # finally remove space
    text = text.split()
    
    return text

#%% load models
llda = tp.load_model("model/llda.bin") # the llda model
key_value = pd.read_pickle("model/key_value.pkl") # each word keyvalue
key_vec = pd.read_pickle('model/key_vec.pkl') # data to vector
key_word = pd.read_pickle('model/key_word.pkl') # data to key words
w2v = fasttext.load_model("model/fasttext.bin") # word to vector, fasttext

# load data
data = pd.read_csv('data/content.csv')

#%% keyword to key vector and key words
def doc_to_key(doc):
    # key value and word vector
    keyvalue = pd.DataFrame([key_value.get(word, 0.1) for word in doc])
    word_vectors = pd.DataFrame([w2v.get_word_vector(word) for word in doc])
     
    # The key value is multiplied by the word vectors row by row
    key_matrix = pd.DataFrame(keyvalue.values * word_vectors.values)
    
    # key vector
    key_vector = key_matrix.sum(axis = 0)
    
    # get unique index
    unique_indices, inverse_indices = np.unique(doc, return_inverse=True)
    
    # add the value with same index 
    aggregated_values = np.bincount(inverse_indices, weights=keyvalue[0])
    
    # merge the result
    key_value_sort = pd.DataFrame(np.column_stack((unique_indices, aggregated_values)))
    
    # the keywords 
    keywords = key_value_sort.sort_values(by=1, ascending=False)
    key_words = keywords.iloc[:10, 0].reset_index(drop=True)
    
    return key_vector, key_words

#%% the pred function
class pred:
    def __init__(self, text):
        self.text = text
        self.pre_process()
        self.llda()
        self.k2v()
    
    def pre_process(self):
        # word segmentate
        self.seg = pre_process(self.text)
        
    def llda(self):
        # recommand department
        corpus = Corpus()
        corpus.add_doc(words = self.seg)
        topic_dist, ll = llda.infer(corpus)
        pred = pd.DataFrame(topic_dist[0].get_topics())
        pred[0] = [llda.topic_label_dict[i] for i in pred[0]]
        pred.columns = ['科別', '機率']
        
        self.pred = pred
        self.ll = ll
        
    def k2v(self):
        # recommand similarity question
        user_key = doc_to_key(self.seg)
        # to keys
        user_vec = user_key[0]
        self.user_key_words = user_key[1]
        # keyword preprocess
        dot_products = np.dot(key_vec, user_vec)
        user_norm = np.linalg.norm(user_vec)
        matrix_norms = np.linalg.norm(key_vec, axis=1)
        cos_sim = dot_products / (user_norm * matrix_norms)
        near_index = np.array(cos_sim).argsort()[::-1]
        
        self.cos_sim = cos_sim
        self.near_index = near_index
    
    def print_result(self):
        sim_question = pd.DataFrame()
        sim_question["科別"] = [data.loc[near, "科別"] for near in self.near_index[:10]]
        
        sim_question["問題主旨"] = [data.loc[near, "問題主旨"] for near in self.near_index[:10]]
        # short the name
        sim_question['問題主旨'] = sim_question['問題主旨'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
        sim_question['問題主旨'] = [f'[{sim_question["問題主旨"][i]}](https://sp1.hso.mohw.gov.tw/doctor/All/ShowDetail.php?q_no={data.loc[self.near_index[i], "q_no"]})' for i in range(10)]
        
        sim_question["關鍵字"] = [", ".join(key_word.iloc[near]) for near in self.near_index[:10]]
        
        sim_question["相似度"] = [self.cos_sim[near] for near in self.near_index[:10]]

        result = {
            "seg": self.seg, 
            "keywords": self.user_key_words,
            "pred": self.pred[['科別', '機率']],
            "sim_question": sim_question
            }
        
        return result

#%% GUI
#
#
#

#%%
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import dash_table

# 初始化 Dash 應用
app = dash.Dash(__name__)

# 定義應用佈局
app.layout = html.Div([
    html.H1("台灣 e 院推薦系統"),
    html.P([
        "資料來自 ",
        html.A("台灣 e 院", href="https://sp1.hso.mohw.gov.tw/doctor/Index1.php", target="_blank"),
        "，更多技術細節參閱 ", 
        html.A("GitHub Page", href="https://github.com/JinZhouLee/taiwan_e_hospital", target="_blank"),
        "。"
    ]), 
    dcc.Textarea(id="user-input", placeholder="請輸入文字", style={"width": "100%", "height": "150px"}),
    html.Button("提交", id="submit-button", n_clicks=0),
    #html.H3("分詞："),
    #html.Div(id="seg"), 
    #html.Hr(),
    html.H3("關鍵字"),
    html.Div(id="keywords"), 
    html.Hr(),
    html.H3("推薦科別："),
    html.Div(id="pred"), 
    html.Hr(),
    html.H3("相似問題："),
    html.Div(id="sim_question") 
])


# 定義回調函數
@app.callback(
    [
     # Output("seg", "children"),
     Output("keywords", "children"),
     Output("pred", "children"), 
     Output("sim_question", "children")],
    Input("submit-button", "n_clicks"),
    State("user-input", "value")
)

def update_output(n_clicks, value):
    if n_clicks > 0 and value:
        # run pred function 
        result = pred(value).print_result()
        
        # show segmentate
        #seg = ', '.join(result['seg'])
        keywords = ', '.join(result['keywords'])
        
        # show pred
        # round the output
        result['pred']['機率'] = np.round(result["pred"]['機率'], 2).astype(str)
        pred_table = dash_table.DataTable(data = result["pred"].to_dict('records'))
        
        # show sim question
        # round the output
        result['sim_question']['相似度'] = np.round(result['sim_question']['相似度'], 2).astype(str)
        sim_question = dash_table.DataTable(
            data = result["sim_question"].to_dict('records'), 
            columns = [{"name": col, "id": col, "presentation": "markdown"} for col in result["sim_question"].columns]
            )
        
        return keywords, pred_table, sim_question
    return None, None, None

# 啟動應用
if __name__ == "__main__":
    app.run(debug=True)

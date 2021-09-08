import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import os
import base64 
import altair as alt
# from st_aggrid import AgGrid
from st_aggrid import AgGrid, GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

st.set_page_config(layout="wide")
url="https://en.wikipedia.org/wiki/2021%E2%80%9322_United_Rugby_Championship"
df=pd.read_html(url, header=0)[9]
st.write(df)
import streamlit as st
from PIL import Image

st.set_page_config(page_title="主页", page_icon="👋",layout='wide')
st.title('香港Airbnb探索性数据分析')
img=Image.open("data/image-airbnb.jpg")
st.image(img)


st.divider()
st.markdown('作者：同济大学 信息管理与信息系统专业 李佳佳')
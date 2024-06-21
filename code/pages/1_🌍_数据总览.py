import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

st.set_page_config(page_title="数据总览", page_icon="🌍",layout="wide")
file_path = 'data/Airbnb/hongkong/listings/202403listings.csv'
file_path2 = 'data/preprocessed/exhibit.csv'
price_path = 'data/preprocessed/export_daily_price.csv'

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

df_listings = load_data(file_path)
df_calendar = load_data(file_path2)

st.markdown('''
## 样本介绍 

在进行Airbnb的分析时我选择了非随机抽样方法中的判断抽样（judgment sampling）。判断抽样是一种基于研究者专业判断和知识来选择样本的非概率抽样方法。在本次分析中，我选择了2023年6月到2025年2月的香港Airbnb数据。

### 选择判断抽样的理由

1. **时效性与现实意义**：在2022年5月24日，Airbnb宣布将暂停支持中国境内游房源、体验及相关预订，退出了中国市场。为了确保数据的时效性和分析的现实意义，我选择了香港的Airbnb数据。香港作为一个国际旅游热点，能提供更多日期较新且有效的数据样本。

2. **数据可获得性**：本项目的数据集来源于Inside Airbnb网站，由于国内Airbnb数据在Inside Airbnb网站上已经被清空，并且较新的数据难以获取，因此选择香港的Airbnb数据更加实际和可行。Inside Airbnb网站提供了全面且最新的香港Airbnb数据，保障了数据分析的完整性和可靠性。

3. **可借鉴性**：香港民宿市场与国内市场在文化和市场结构上有一定的相似性。通过对香港Airbnb数据集的分析，不仅可以为香港的旅客和房东提供有价值的见解，还能为国内的民宿市场提供参考和借鉴，助力市场优化和发展。
''')
st.divider()
st.markdown('## 数据总览')
st.markdown('### 房源信息')
st.write('描述房源编号、地理位置、房源内设施等，行数: '+str(df_listings.shape[0])+', 列数: '+str(df_listings.shape[1]))
st.write(df_listings.head())

st.markdown('### 房源空闲情况信息')
st.write('描述日期、当日价格、当日是否可用等，行数: 2213360, 列数: 7')
st.write(df_calendar.head())
st.write("来源：https://insideairbnb.com/")
st.divider()
st.markdown('## 缺失值可视化')
st.markdown('### 房源信息')
fig1, ax1 = plt.subplots(figsize=(10, 6))  # 调整图的比例
msno.matrix(df_listings, ax=ax1, sparkline=False)
st.pyplot(fig1)
st.markdown('### 房源空闲情况信息')
fig2, ax2 = plt.subplots(figsize=(10, 6))  
msno.matrix(df_calendar, ax=ax2, sparkline=False)
st.pyplot(fig2)
st.divider()
st.markdown('## 异常值分析')
df_listings['price'] = df_listings['price'].replace('[\$,]', '', regex=True).astype(float)
num_columns = ['price','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value']

df_numeric = df_listings[num_columns].dropna()
num_plots_per_row = 4

for i in range(0, len(num_columns), num_plots_per_row):
    cols = st.columns(num_plots_per_row)
    for j, col in enumerate(cols):
        if i + j < len(num_columns):
            column = num_columns[i + j]
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.boxenplot(y=df_numeric[column], ax=ax)
            ax.set_title(f'Boxenplot of {column}')
            ax.set_yticks([df_numeric[column].min(), df_numeric[column].max()])
            col.pyplot(fig)

st.divider()
st.markdown('## 特征工程')
st.markdown('''
由于价格数据集记录的是每个房源在未来365天内的价格信息（在聚合多个文件后总计覆盖了1.5年，共548天的数据），直接进行分析较为不便。因此，我根据日期、季节等因素进行了特征工程，生成了以下变量：
                        
### 特征工程生成变量

1. **每种房型每日平均价格**：根据日期和房型计算了每日的平均价格。

2. **每个房源每月平均价格**：根据房源ID和每日价格计算了每个房源的每月平均价格。

3. **房源季节使用率**：将每个房源每日是否可用的状态按照季度汇总，转换为了每个季度的使用率。
''')

st.divider()
st.markdown('作者：同济大学 信息管理与信息系统专业 李佳佳')
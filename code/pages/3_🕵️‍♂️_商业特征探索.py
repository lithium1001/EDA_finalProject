import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pywaffle import Waffle
from sklearn.cluster import KMeans

st.set_page_config(page_title="商业特征探索", page_icon="🕵️‍♂️",layout="wide")

file_path = 'data/Airbnb/hongkong/listings/202403_sum_listings.csv'
daily_price_path = 'data/preprocessed/export_daily_price.csv'
availability_path = 'data/preprocessed/export_seasonal_availability.csv'
monthly_price_path = 'data/preprocessed/export_filtered.csv'

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

df_price_ori=load_data(daily_price_path)
df_ori=load_data(file_path)
df_a=load_data(availability_path)
df_m=load_data(monthly_price_path)


st.markdown('## 价格特征')
st.markdown('### 各房型日均价格波动')
df_price_ori['date'] = pd.to_datetime(df_price_ori['date'])
df_price= df_price_ori[df_price_ori['date'] < '2024-03-24']
price_pivot = df_price.pivot(index='date', columns='room_type', values='avg_price').reset_index()

fig_price=go.Figure([
   go.Scatter(x=price_pivot["date"], y=price_pivot["Entire home/apt"], xaxis="x", yaxis="y1", name="Entire home/apt"),
    go.Scatter(x=price_pivot["date"], y=price_pivot["Hotel room"], xaxis="x", yaxis="y2", name="Hotel room"),
    go.Scatter(x=price_pivot["date"], y=price_pivot["Private room"], xaxis="x", yaxis="y3", name="Private room"),
    go.Scatter(x=price_pivot["date"], y=price_pivot["Shared room"], xaxis="x", yaxis="y4", name="Shared room"),
])
fig_price.update_layout(
    width=1200, 
    height=700,
    hoversubplots="overlaying",
    hovermode="x unified",  # or "x unified" without dashed line    
    grid=dict(
        rows=4, 
        columns=1,  # the grid cells (row, col): [[(0,0), (0,1)], [(1,0), (1,1)]
        subplots=[["xy"],  # 共享 x 轴
                  ["xy2"], 
                  ["xy3"],
                  ["xy4"]],
        xgap=0.1, 
        ygap=0.2,
        
    ),
    xaxis=dict(domain=[0, 1]),
    yaxis=dict(domain=[0.75, 1], title="Entire home/apt"),
    yaxis2=dict(domain=[0.5, 0.75], title="Hotel room"),
    yaxis3=dict(domain=[0.25, 0.5], title="Private room"),
    yaxis4=dict(domain=[0, 0.25], title="Shared room")
)

st.plotly_chart(fig_price)

st.markdown('### 各房型月均价格分布')
count_less_than_10 = df_m[df_m['avg_price'] < 100].shape[0]
df_m = df_m[df_m['avg_price'] >= 100]

fig_rp = px.violin(df_m, x='room_type', y='avg_price', color='room_type', points='all',box=True,animation_frame='month')
fig_rp.update_layout(width=1200, height=700)
st.plotly_chart(fig_rp)


df_grouped = df_m.groupby(['room_type', 'neighbourhood', 'month']).agg(
    avg_price=('avg_price', 'mean'),
    num_listings=('avg_price', 'size')
).reset_index()


fig3 = px.scatter(
    df_grouped,
    x='neighbourhood',
    y='avg_price',
    color='room_type',
    size='num_listings',
    size_max=100,
    animation_frame='month',
)
fig3.update_layout(width=1200, height=700)
st.markdown('### 各行政区不同房型的每月均价及房源数量')
st.markdown("横轴表示行政区，纵轴表示每月均价，气泡大小表示当月房源数量，颜色表示房型。")
st.plotly_chart(fig3)
st.divider()
st.markdown('## 房东特征')

# 按host_name进行分组，并计算price和minimum_nights的均值
df_grouped = df_ori.groupby('host_name').agg({'price': 'mean', 'minimum_nights': 'mean'}).reset_index()

# 去掉有缺失值的行
df_grouped = df_grouped.dropna()

# 按百分位数将price映射到1-100的等级
df_grouped['price_percentile'] = df_grouped['price'].rank(pct=True) * 100

# 绘制手肘图
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, n_init=10)
    kmeans.fit(df_grouped[['price_percentile', 'minimum_nights']])
    wcss.append(kmeans.inertia_)

fig_elbow = go.Figure()
fig_elbow.add_trace(go.Scatter(x=list(range(1, 11)), y=wcss, mode='lines+markers'))
fig_elbow.update_layout(xaxis_title='聚类数', yaxis_title='WCSS')
st.markdown('### 基于房源均价和最短入住天数的房东聚类分析')
st.markdown('#### 使用肘部法确定最佳聚类数')
st.markdown("肘部法（Elbow Method）通过绘制不同K值的总误差平方和（SSE）曲线并寻找“SSE明显下降后趋于平缓的拐点”，确定K-means聚类的最佳聚类数。")
st.plotly_chart(fig_elbow)

# 确定最佳聚类数
optimal_clusters = 3

# 进行KMeans聚类
kmeans = KMeans(n_clusters=optimal_clusters, n_init=10)
df_grouped['cluster'] = kmeans.fit_predict(df_grouped[['price_percentile', 'minimum_nights']])

# 可视化聚类结果
fig_clusters = go.Figure()

# 添加散点数据
fig_clusters.add_trace(go.Scatter(
    x=df_grouped['price_percentile'],
    y=df_grouped['minimum_nights'],
    mode='markers',
    marker=dict(
        size=10,  # 固定散点大小
        color=df_grouped['cluster'],  # 使用 'cluster' 作为颜色
        colorscale=px.colors.qualitative.Plotly,  # 设置颜色映射
        colorbar=dict(title='Cluster', showticklabels=False),  # 隐藏颜色条
        showscale=False  # 不显示颜色条
    ),
    text=df_grouped['host_name'],  # 气泡上的文本
    hoverinfo='text'
))
st.write('#### 房东聚类结果')
# 更新图表布局
fig_clusters.update_layout(
    xaxis=dict(title='价格百分位数'),
    yaxis=dict(title='最小入住天数'),
    width=1200,
    height=700
)

st.plotly_chart(fig_clusters)
st.divider()
st.markdown('### 房东与房源特征分析（基于持有房源数量分类）')
# 预处理数据
data = df_ori.groupby('host_name').size().reset_index(name='count')

# 获取唯一的房源数量值
unique_counts = sorted(data['count'].unique())

# 添加滑条来选择房源数量范围
min_count, max_count = st.select_slider(
    '选择房源数量范围',
    options=unique_counts,
    value=(unique_counts[0], unique_counts[-1])
)

# 过滤数据以匹配选择的范围
filtered_data = data[(data['count'] >= min_count) & (data['count'] <= max_count)]
num_hosts = len(filtered_data)

# 根据过滤后的数据选择相关的房源
filtered_hosts = filtered_data['host_name'].tolist()
filtered_ori = df_ori[df_ori['host_name'].isin(filtered_hosts)]

# 添加单选按钮以选择图表类型
chart_type = st.radio(
    '选择图表类型',
    ( '直方图','树形图', '桑基图'),
    horizontal=True
)

@st.experimental_fragment
def draw_treemap(data):
    fig = px.treemap(
        data, 
        title=None,
        path=['host_name'], 
        values='count', 
        color='host_name',
        labels={'count': '房源个数'}
    )
    
    fig.update_traces(
        texttemplate='%{label}<br>%{value}', 
        textinfo='label+text+value',
        textposition='middle center',
        textfont_size=20,
                
    )
    fig.update_layout(width=1300, height=800 )
    return fig

@st.experimental_fragment
def draw_histogram(data):
    fig = px.histogram(
        data, 
        x='count', 
        labels={'count': '房源个数', 'frequency': '频率'},
        nbins=len(unique_counts),
    )
    
    fig.update_layout(width=600, height=600, )
    return fig

@st.experimental_fragment
def draw_sankey(data):
    # 过滤掉不包含所有三个字段的记录
    data = data.dropna(subset=['host_name', 'neighbourhood', 'room_type'])

    # 获取所有节点标签
    all_hosts = data['host_name'].unique().tolist()
    all_neighbourhoods = data['neighbourhood'].unique().tolist()
    all_room_types = data['room_type'].unique().tolist()
    
    node_labels = all_hosts + all_neighbourhoods + all_room_types
    
    # 为节点设置颜色
    color_map = px.colors.qualitative.Plotly
    node_colors = [color_map[i % len(color_map)] for i in range(len(node_labels))]
    
    # 构建源和目标索引
    source_indices = []
    target_indices = []
    for _, row in data.iterrows():
        host_idx = node_labels.index(row['host_name'])
        neighbourhood_idx = node_labels.index(row['neighbourhood'])
        room_type_idx = node_labels.index(row['room_type'])
        
        # 确保连接顺序是 host -> neighbourhood -> room_type
        source_indices.extend([host_idx, neighbourhood_idx])
        target_indices.extend([neighbourhood_idx, room_type_idx])
    
    link_values = [1] * len(source_indices)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=link_values,
            color='rgba(82,139,139, 0.3)'  # 设置连接线的颜色和透明度
        ))])

    fig.update_layout(
        font_size=20,
        width=1300,  # 增加图表宽度
        height=800,  # 增加图表高度
    )
    return fig

config = dict({'displayModeBar': False})
if not filtered_data.empty:
    if chart_type == '树形图':        
        st.markdown(f"#### {num_hosts}个房东的房源数量分布")
        fig = draw_treemap(filtered_data)
    elif chart_type == '直方图':        
        st.markdown(f"#### {num_hosts}个房东的房源数量分布")
        fig = draw_histogram(filtered_data)
    else:
        st.markdown(f"#### {num_hosts}个房东的房源桑基图 - 行政区与房型分布")
        fig = draw_sankey(filtered_ori)
    
    st.plotly_chart(fig, **{'config': config})
else:
    st.write("No data available.")

st.divider()
st.markdown('## 房客特征')

df_a['occupied_days'] = df_a['total_days'] - df_a['available_days']
avg_data = df_a.groupby(['room_type', 'season']).agg({
    'available_days': 'mean',
    'occupied_days': 'mean'
}).reset_index()

season_order = ['Spring', 'Summer', 'Autumn', 'Winter']

# Define colors for each room type
room_colors = {
    'Entire home/apt': ['#0068C9', '#d3d3d3'],  # Blue
    'Private room': ['#83C9FF', '#d3d3d3'],    # Light blue
    'Shared room': ['#f95743', '#d3d3d3'],     # Red
    'Hotel room': ['#FFABAB', '#d3d3d3']       # Light red
}

# Prepare the Streamlit layout with an additional index column
st.write("### 各季各房型使用率")
# Plot each room type and season combination in the appropriate cell
@st.cache_data
def waffle(avg_data, room_colors, season_order):
    for room_type in avg_data['room_type'].unique():
        col = st.columns([0.4, 1, 1, 1, 1])  # Adjust the width of the first column
        col[0].write(room_type)
        for j, season in enumerate(season_order):
            filtered_data = avg_data[(avg_data['room_type'] == room_type) & (avg_data['season'] == season)]
            if not filtered_data.empty:
                available_days = round(filtered_data['available_days'].values[0] / 30)
                occupied_days = round(filtered_data['occupied_days'].values[0] / 30)
                total_days = available_days + occupied_days
                occupancy_rate = available_days / total_days if total_days != 0 else 0
                
                data_dict = {'Free Days': available_days, 'Occupied Days': occupied_days}
                
                # Create a waffle chart
                fig = plt.figure(
                    FigureClass=Waffle, 
                    rows=5,  # 5 rows to accommodate 30 blocks
                    columns=6,  # 6 columns to accommodate 30 blocks
                    values=data_dict,
                    colors=room_colors[room_type],  # Use the color for the specific room type
                    legend={'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 8},
                    title={'label': f'{season}', 'loc': 'center', 'fontsize': 12},
                    figsize=(4, 4),  # Size of each waffle chart
                    starting_location='NW',
                    vertical=True,
                )
                fig.axes[0].get_legend().remove()

                # Add the occupancy rate text
                fig.text(
                    x=0.5,
                    y=0.5,
                    s=f"{occupancy_rate:.1%}",
                    ha="center",
                    va="center",
                    rotation=0,
                    fontsize=45,
                    color='gray',
                    # weight='bold'
                    
                )
                
                col[j + 1].pyplot(fig)

waffle(avg_data, room_colors, season_order)


merged_data = pd.merge(df_a, df_ori, left_on='listing_id', right_on='id',how='inner')

merged_data.drop(columns=['neighbourhood_group'], inplace=True)

annual_data = merged_data.groupby(['listing_id', 'season']).agg(
    total_available_days=('available_days', 'sum'),
    total_days=('total_days', 'sum'),
    price=('price', 'mean'),
    number_of_reviews=('number_of_reviews', 'sum'),
    minimum_nights=('minimum_nights', 'mean'),
    availability_365=('availability_365', 'mean'),
    neighbourhood=('neighbourhood', 'first'),
    room_type=('room_type_x', 'first')
).reset_index()

# 计算全年使用率
annual_data['occupancy_ratio'] = 1 - (annual_data['total_available_days'] / annual_data['total_days'])

# 独热编码房型和行政区
encoded_data = pd.get_dummies(annual_data, columns=['neighbourhood', 'room_type','season'])

# 保留相关字段
relevant_columns = [
    'occupancy_ratio', 'price', 'number_of_reviews', 'minimum_nights', 'availability_365'
] + [col for col in encoded_data.columns if col.startswith('neighbourhood_') or col.startswith('room_type_') or col.startswith('season_')]
final_data = encoded_data[relevant_columns]

# 计算相关性矩阵
corr_matrix = final_data.corr()

# 使用Plotly绘制热图
fig4 = px.imshow(corr_matrix, 
                text_auto=True, 
                color_continuous_scale='RdBu_r', 
                )
fig4.update_layout(width=900, height=900)
st.markdown('### 房客关键特征相关性矩阵')
st.plotly_chart(fig4)

st.divider()
st.markdown('作者：同济大学 信息管理与信息系统专业 李佳佳')
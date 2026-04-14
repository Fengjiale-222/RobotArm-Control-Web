import streamlit as st
import numpy as np
import plotly.graph_objects as go
import socket
import json
import paho.mqtt.client as mqtt

# ==========================================
# 0. 页面全局配置与移动端适配 CSS
# ==========================================
st.set_page_config(page_title="模块化机械臂分拣系统", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            padding-bottom: 1rem !important;
        }
        h1 {
            font-size: 1.5rem !important;
            padding-bottom: 5px !important;
        }
        .stSlider {
            padding-bottom: 15px !important;
        }
        button[data-baseweb="tab"] {
            font-size: 0.85rem !important;
            padding: 10px 4px !important;
        }
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. 🔐 系统安全登录拦截器
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #1E88E5;'>🛡️ 机械臂中控台安全验证</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>警告：非授权人员禁止操作物理实验设备</p>", unsafe_allow_html=True)
        
        st.write("")
        pwd = st.text_input("请输入访问密码：", type="password", placeholder="请输入授权密码...")
        
        if st.button("🚀 登 入 系 统", type="primary", width="stretch"):
            if pwd == "123456": 
                st.session_state.logged_in = True
                st.rerun() 
            else:
                st.error("❌ 密码错误，请重新输入或联系系统管理员。")
    st.stop() 

# ==========================================
# (主系统)
# ==========================================
st.markdown("""
    <h1 style='text-align: center; color: #1E88E5; padding-bottom: 20px;'>
        🦾 基于 LVGL 与视觉识别的模块化机械臂分拣系统
    </h1>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 机械臂物理参数
# ==========================================
class RobotParams:
    L1 = 0.08; L2 = 0.155; L3 = 0.11; L4 = 0.11
    d2 = -0.0165; d3 = -0.0165
p = RobotParams()

Q_SAFE = np.array([0.0, -24.0, -41.0, 75.0])
HOVER_OFFSET = 0.05  

# ==========================================
# 3. 运动学算法与状态记忆
# ==========================================
if "curr_x" not in st.session_state: st.session_state.curr_x = 0.150
if "curr_y" not in st.session_state: st.session_state.curr_y = 0.000
if "curr_z" not in st.session_state: st.session_state.curr_z = 0.100
if "claw_closed" not in st.session_state: st.session_state.claw_closed = False

def calculate_4dof_ik(x, y, z):
    try:
        r_dist = np.sqrt(x**2 + y**2)
        if r_dist < 0.005: return None, "靠近中心死区"
        j1_math = np.arctan2(y, x)
        r_j4 = r_dist; z_j4 = z + p.L4 - p.L1
        A = np.sqrt(p.L2**2 + p.d2**2); alpha1 = np.arctan2(p.d2, p.L2)
        B = np.sqrt(p.L3**2 + p.d3**2); alpha2 = np.arctan2(p.d3, p.L3)
        D_sq = r_j4**2 + z_j4**2; D = np.sqrt(D_sq)
        if D > (A + B) or D < abs(A - B): return None, "超出限界"
        cos_u2 = np.clip((D_sq - A**2 - B**2) / (2 * A * B), -1, 1)
        u2 = -np.arccos(cos_u2) 
        u1 = np.arctan2(z_j4, r_j4) - np.arctan2(B * np.sin(u2), A + B * np.cos(u2))
        m1, m2, m3 = j1_math, u1 - alpha1, u2 + alpha1 - alpha2
        m4 = -np.pi/2 - m2 - m3
        return np.array([np.degrees(m1), np.degrees(np.pi/2 - m2), np.degrees(m3 + np.pi/2), np.degrees(-m4)]), "OK"
    except Exception as e:
        return None, "Error"

def forward_kinematics_3d(hw_angles):
    j1, j2, j3, j4 = np.radians(hw_angles)
    m1, m2, m3 = j1, np.pi/2 - j2, j3 - np.pi/2
    pts = [[0, 0, 0], [0, 0, p.L1]] 
    x3 = np.cos(m1) * (p.L2 * np.cos(m2) - p.d2 * np.sin(m2))
    y3 = np.sin(m1) * (p.L2 * np.cos(m2) - p.d2 * np.sin(m2))
    z3 = p.L1 + p.L2 * np.sin(m2) + p.d2 * np.cos(m2)
    pts.append([x3, y3, z3])
    m23 = m2 + m3
    x4 = x3 + np.cos(m1) * (p.L3 * np.cos(m23) - p.d3 * np.sin(m23))
    y4 = y3 + np.sin(m1) * (p.L3 * np.cos(m23) - p.d3 * np.sin(m23))
    z4 = z3 + p.L3 * np.sin(m23) + p.d3 * np.cos(m23)
    pts.append([x4, y4, z4]); pts.append([x4, y4, z4 - p.L4]) 
    return np.array(pts)

# ==========================================
# 4. 3D 动画引擎
# ==========================================
def plot_dynamic_trajectory(points_list, direct_move=False):
    fig = go.Figure()
    R_max = p.L2 + p.L3
    u, v = np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi / 2, 15) 
    fig.add_trace(go.Surface(x=R_max * np.outer(np.cos(u), np.sin(v)), y=R_max * np.outer(np.sin(u), np.sin(v)), z=p.L1 + R_max * np.outer(np.ones(np.size(u)), np.cos(v)), opacity=0.1, colorscale='Blues', showscale=False, name='工作空间'))

    key_angles_seq = [Q_SAFE]
    
    if len(points_list) > 0:
        if direct_move:
            valid_angles = [calculate_4dof_ik(pt[0], pt[1], pt[2])[0] for pt in points_list]
            key_angles_seq = [a for a in valid_angles if a is not None]
        else:
            for pt in points_list:
                ans_h, _ = calculate_4dof_ik(pt[0], pt[1], pt[2] + HOVER_OFFSET)
                ans_t, _ = calculate_4dof_ik(pt[0], pt[1], pt[2])
                if ans_h is not None and ans_t is not None:
                    key_angles_seq.extend([ans_h, ans_t, ans_h, Q_SAFE])

    if len(key_angles_seq) > 0:
        tcp_pts = [forward_kinematics_3d(q)[-1] for q in key_angles_seq]
        if len(key_angles_seq) > 1:
            fig.add_trace(go.Scatter3d(x=[pt[0] for pt in tcp_pts], y=[pt[1] for pt in tcp_pts], z=[pt[2] for pt in tcp_pts], mode='lines+markers', line=dict(color='red', width=4, dash='dash'), marker=dict(size=6, color='red'), name='预计轨迹'))
        
        init_pts = forward_kinematics_3d(key_angles_seq[0])
        fig.add_trace(go.Scatter3d(x=init_pts[:,0], y=init_pts[:,1], z=init_pts[:,2], mode='lines+markers', line=dict(color='darkblue', width=10), marker=dict(size=8, color=['black', 'orange', 'orange', 'orange', 'green']), name='数字孪生'))

        frames, frame_idx = [], 0
        for i in range(len(key_angles_seq) - 1):
            q_start, q_end = key_angles_seq[i], key_angles_seq[i+1]
            steps = 10
            for j in range(steps):
                s = (j / (steps - 1)) ** 2 * (3 - 2 * (j / (steps - 1))) 
                pts = forward_kinematics_3d(q_start + (q_end - q_start) * s)
                frames.append(go.Frame(data=[go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2])], name=str(frame_idx), traces=[2]))
                frame_idx += 1

        fig.frames = frames
        fig.update_layout(updatemenus=[dict(type="buttons", showactive=False, buttons=[dict(label="▶ 播放实体预演", method="animate", args=[None, {"frame": {"duration": 30, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True, "mode": "immediate"}])], x=0.05, y=0.1)], scene=dict(xaxis=dict(range=[-0.3, 0.3], title='X (m)'), yaxis=dict(range=[-0.3, 0.3], title='Y (m)'), zaxis=dict(range=[0, 0.4], title='Z (m)'), aspectmode='cube'), margin=dict(l=0, r=0, b=0, t=0), height=500, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig

# ==========================================
# 5. 侧边栏：核心控制链路
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/robot-3.png", width=80) 
    st.header("⚙️ 核心控制链路")
    mqtt_broker = st.text_input("MQTT 服务器地址", value="broker.emqx.io")
    mqtt_topic_pub = st.text_input("下发指令主题", value="biyeshe_robot_arm_2026/target")
    
    if st.button("🔌 测试云端连通性", width="stretch"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            if s.connect_ex((mqtt_broker, 1883)) == 0:  
                st.success(f"✅ 已成功连接至 {mqtt_broker}")
            else: 
                st.error("❌ 无法连接，请检查网络")
            s.close()
        except: 
            st.error("❌ 网络异常")
            
    st.divider()
    
    if st.button("🚪 安全退出系统", width="stretch"):
        st.session_state.logged_in = False
        st.rerun()
        
    st.info("💡 **系统提示**\n\n请在右侧主工作区的不同选项卡中，切换机械臂的运行模式。")

def send_mqtt_payload(payload_dict):
    try:
        json_str = json.dumps(payload_dict, separators=(',', ':'))
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect(mqtt_broker, 1883, 60)
        client.publish(mqtt_topic_pub, json_str)
        client.disconnect()
        st.toast("✅ 指令已极速下发至底层主控！")
    except Exception as e:
        st.error(f"❌ 发送失败：{e}")

def check_reachable(x, y, z, hover=True):
    if hover:
        ans_h, _ = calculate_4dof_ik(x, y, z + HOVER_OFFSET)
        ans_t, _ = calculate_4dof_ik(x, y, z)
        return (ans_h is not None and ans_t is not None)
    else:
        return calculate_4dof_ik(x, y, z)[0] is not None

# ==========================================
# 6. 主工作区
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 模式一：单点门字型抓取", 
    "🏭 模式二：双点流水线搬运", 
    "✨ 模式三：阵列式批量作业", 
    "🕹️ 模式四：空间摇杆精准示教"
])

# --- Tab 1: 单点门字型 ---
with tab1:
    st.markdown("#### 🎯 单点目标精确定位与抓取")
    c1, c2 = st.columns([1, 1.8])
    with c1:
        st.write("##### 📍 目标点坐标设定")
        x1 = st.slider("X 轴坐标 (m)", -0.25, 0.25, 0.15, step=0.001, format="%.3f", key="t1_x")
        y1 = st.slider("Y 轴坐标 (m)", -0.25, 0.25, 0.05, step=0.001, format="%.3f", key="t1_y")
        z1 = st.slider("Z 轴高度 (m)", -0.05, 0.25, 0.05, step=0.001, format="%.3f", key="t1_z")
        
        is_ok = check_reachable(x1, y1, z1)
        st.divider()
        
        if is_ok:
            if st.button("🚀 编译并下发单点任务", type="primary", width="stretch"):
                send_mqtt_payload({"mode": 1, "target": {"x": x1, "y": y1, "z": z1, "claw": 1}})
        else:
            st.error("⚠️ 坐标超出设置范围，无法下发！")
            
    with c2: 
        if is_ok:
            st.plotly_chart(plot_dynamic_trajectory([(x1, y1, z1)]), width="stretch", key="chart_t1_ok")
        else:
            st.plotly_chart(plot_dynamic_trajectory([]), width="stretch", key="chart_t1_err")

# --- Tab 2: 双点搬运 ---
with tab2:
    st.markdown("#### 🏭 双点流水线：自动完成「取料 -> 放料 -> 复位」闭环逻辑")
    c1, c2 = st.columns([1, 1.8])
    with c1:
        with st.expander("📦 取料点参数设定", expanded=True):
            p1x = st.slider("X1", -0.25, 0.25, 0.15, step=0.001, format="%.3f", key="t2_x1")
            p1y = st.slider("Y1", -0.25, 0.25, 0.05, step=0.001, format="%.3f", key="t2_y1")
            p1z = st.slider("Z1", -0.05, 0.25, 0.05, step=0.001, format="%.3f", key="t2_z1")
        with st.expander("📤 放料点参数设定", expanded=True):
            p2x = st.slider("X2", -0.25, 0.25, 0.15, step=0.001, format="%.3f", key="t2_x2")
            p2y = st.slider("Y2", -0.25, 0.25, -0.05, step=0.001, format="%.3f", key="t2_y2")
            p2z = st.slider("Z2", -0.05, 0.25, 0.05, step=0.001, format="%.3f", key="t2_z2")
            
        is_all_ok = check_reachable(p1x, p1y, p1z) and check_reachable(p2x, p2y, p2z)
        st.divider()
        
        if is_all_ok:
            if st.button("🚀 编译并下发流水线任务", type="primary", width="stretch"):
                send_mqtt_payload({"mode": 2, "t1": {"x": p1x, "y": p1y, "z": p1z, "claw": 1}, "t2": {"x": p2x, "y": p2y, "z": p2z, "claw": 0}})
        else:
            st.error("⚠️ 流水线坐标超出设置范围，无法下发！")
            
    with c2: 
        if is_all_ok:
            st.plotly_chart(plot_dynamic_trajectory([(p1x, p1y, p1z), (p2x, p2y, p2z)]), width="stretch", key="chart_t2_ok")
        else:
            st.plotly_chart(plot_dynamic_trajectory([]), width="stretch", key="chart_t2_err")

# --- Tab 3: 阵列式多次作业 ---
with tab3:
    st.markdown("#### ✨ 阵列式复杂路径规划 (支持 LVGL 视觉坐标连续传入)")
    col1, col2 = st.columns([1, 1.8])
    with col1:
        # ⚠️ 修复点：已将最小节点数强制限制为 3
        num_points = st.number_input("设定流水线连续作业的节点数量：", min_value=3, max_value=8, value=3, step=1)
        st.divider()
        points_data = []
        is_all_ok = True
        
        for i in range(num_points):
            with st.expander(f"📍 阵列节点 {i+1}", expanded=(i<2)):
                px = st.slider("X", -0.25, 0.25, 0.15, step=0.001, format="%.3f", key=f"mx{i}")
                py = st.slider("Y", -0.25, 0.25, 0.05 * (1 if i%2==0 else -1), step=0.001, format="%.3f", key=f"my{i}")
                pz = st.slider("Z", -0.05, 0.25, 0.05, step=0.001, format="%.3f", key=f"mz{i}")
                claw = st.checkbox("🛠️ 到达后闭合爪子", value=(i%2==0), key=f"mc{i}") 
                
                if check_reachable(px, py, pz):
                    points_data.append({"x": px, "y": py, "z": pz, "claw": 1 if claw else 0})
                else:
                    st.error(f"⚠️ 节点 {i+1} 已超出设置范围！")
                    is_all_ok = False
                    
        st.divider()
        if is_all_ok:
            if st.button(f"🚀 编译并连续执行 {num_points} 个列阵动作", type="primary", width="stretch"):
                send_mqtt_payload({"mode": 3, "count": num_points, "pts": points_data})
        else:
            st.error("⚠️ 必须修正所有超出范围的坐标后，方可下发任务。")

    with col2:
        if is_all_ok:
            pts_tuple = [(p['x'], p['y'], p['z']) for p in points_data]
            st.plotly_chart(plot_dynamic_trajectory(pts_tuple), width="stretch", key="chart_t3_ok")
        else:
            st.plotly_chart(plot_dynamic_trajectory([]), width="stretch", key="chart_t3_err")

# --- Tab 4: 实时摇杆控制 ---
with tab4:
    st.markdown("#### 🕹️ 空间姿态点动示教器 (Jogging Control)")
    col_ctrl, col_status = st.columns([1.2, 1])
    
    with col_ctrl:
        st.write("##### 🎮 XYZ 空间微调摇杆")
        step = st.selectbox("全局步进精度 (Step Size)", [0.005, 0.010, 0.020, 0.050], index=1, format_func=lambda x: f"{x*1000} mm")
        
        def move_axis(axis, d):
            new_x, new_y, new_z = st.session_state.curr_x, st.session_state.curr_y, st.session_state.curr_z
            if axis == 'x': new_x += d
            elif axis == 'y': new_y += d
            elif axis == 'z': new_z += d
            
            if check_reachable(new_x, new_y, new_z, hover=False):
                st.session_state.curr_x, st.session_state.curr_y, st.session_state.curr_z = new_x, new_y, new_z
                send_mqtt_payload({"mode": 4, "target": {"x": new_x, "y": new_y, "z": new_z, "claw": 1 if st.session_state.claw_closed else 0}})
            else:
                st.error("⚠️ 动作已拦截：坐标超出设置范围！")

        c1, c2, c3, c4 = st.columns(4)
        with c2: 
            if st.button("X+ (向前)", width="stretch"): move_axis('x', step)
        with c1: 
            if st.button("Y+ (向左)", width="stretch"): move_axis('y', step)
        with c3: 
            if st.button("Y- (向右)", width="stretch"): move_axis('y', -step)
        with c2: 
            if st.button("X- (向后)", width="stretch"): move_axis('x', -step)
        with c4:
            if st.button("Z+ (向上)", width="stretch"): move_axis('z', step)
            if st.button("Z- (向下)", width="stretch"): move_axis('z', -step)
            
        st.divider()
        if st.button("🛠️ 触发末端执行器 (张开/闭合爪子)", width="stretch"):
            st.session_state.claw_closed = not st.session_state.claw_closed
            send_mqtt_payload({"mode": 4, "target": {"x": st.session_state.curr_x, "y": st.session_state.curr_y, "z": st.session_state.curr_z, "claw": 1 if st.session_state.claw_closed else 0}})

    with col_status:
        st.write("##### 📊 空间位置数字孪生")
        st.metric("实时空间末端坐标 (TCP)", f"X: {st.session_state.curr_x:.3f} | Y: {st.session_state.curr_y:.3f} | Z: {st.session_state.curr_z:.3f}")
        fig = plot_dynamic_trajectory([(st.session_state.curr_x, st.session_state.curr_y, st.session_state.curr_z)], direct_move=True)
        st.plotly_chart(fig, width="stretch", key="chart_t4_ok")
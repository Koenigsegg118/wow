import zipfile
import re
import numpy as np
import torch
import io


class AirCombatTokenizer:
    def __init__(self, ego_id):
        self.ego_id = ego_id
        self.current_frame_data = {}
        # 定义Token的特征维度: [x, y, z, roll, pitch, yaw, u, v, w]
        self.feature_dim = 9

    def parse_acmi_line(self, line):
        """
        解析ACMI的一行数据。
        格式通常为: 801,T=Lon|Lat|Alt|Roll|Pitch|Yaw|U|V|Heading...
        """
        if line.startswith("#"): return None, None  # 跳过注释

        parts = line.split(',')
        obj_id = parts[0]

        # 简单的解析逻辑，实际ACMI可能更复杂，包含变长参数
        # 这里假设我们要提取包含 T= 的遥测数据
        telemetry = {}
        for part in parts[1:]:
            if part.startswith('T='):
                # ACMI格式: T=Lon|Lat|Alt|Roll|Pitch|Yaw
                # 注意：Tacview的数据可能省略某些未变化的字段，这里做简化处理
                values = part[2:].split('|')
                try:
                    # 填充数据，缺失的设为0或保持上一帧(简化演示设为0)
                    telemetry['lon'] = float(values[0]) if len(values) > 0 and values[0] else 0.0
                    telemetry['lat'] = float(values[1]) if len(values) > 1 and values[1] else 0.0
                    telemetry['alt'] = float(values[2]) if len(values) > 2 and values[2] else 0.0
                    telemetry['roll'] = float(values[3]) if len(values) > 3 and values[3] else 0.0
                    telemetry['pitch'] = float(values[4]) if len(values) > 4 and values[4] else 0.0
                    telemetry['yaw'] = float(values[5]) if len(values) > 5 and values[5] else 0.0
                except ValueError:
                    continue

        if not telemetry:
            return None, None

        return obj_id, telemetry

    def world_to_ego_frame(self, ego_data, agent_data):
        """
        【GenAD 核心逻辑迁移】
        将Agent的绝对坐标转换为相对于Ego的相对坐标。
        这对应于论文中的 Instance-Centric 转换。
        """
        # 简化计算：仅计算经纬度高度差，实际工程需转换为局部笛卡尔坐标系(NED)
        d_lon = agent_data['lon'] - ego_data['lon']
        d_lat = agent_data['lat'] - ego_data['lat']
        d_alt = agent_data['alt'] - ego_data['alt']

        # 计算相对姿态 (简化减法)
        d_roll = agent_data['roll'] - ego_data['roll']
        d_pitch = agent_data['pitch'] - ego_data['pitch']
        d_yaw = agent_data['yaw'] - ego_data['yaw']

        # 构建特征向量
        return [d_lon, d_lat, d_alt, d_roll, d_pitch, d_yaw, 0, 0, 0]  # 预留速度位

    def tokenize_frame(self, raw_lines):
        """
        处理一帧的数据，生成 Tensors
        """
        # 1. 更新当前帧所有单位的状态
        for line in raw_lines:
            obj_id, data = self.parse_acmi_line(line)
            if obj_id and data:
                # 在真实场景中需要merge上一帧的数据，因为ACMI只记录变化量
                # 这里为了演示直接覆盖
                if obj_id not in self.current_frame_data:
                    self.current_frame_data[obj_id] = data
                else:
                    self.current_frame_data[obj_id].update(data)

        # 2. 检查Ego是否存在
        if self.ego_id not in self.current_frame_data:
            return None

        ego_data = self.current_frame_data[self.ego_id]

        # 3. 构建 Ego Token (自身状态)
        # Ego Token 通常保留绝对姿态或相对于地面的姿态
        ego_token = [0, 0, ego_data['alt'], ego_data['roll'], ego_data['pitch'], ego_data['yaw'], 0, 0, 0]

        # 4. 构建 Agent Tokens (僚机、敌机、导弹)
        agent_tokens_list = []
        agent_ids = []

        for oid, odata in self.current_frame_data.items():
            if oid == self.ego_id: continue

            # 计算相对特征
            rel_feature = self.world_to_ego_frame(ego_data, odata)
            agent_tokens_list.append(rel_feature)
            agent_ids.append(oid)

        # 5. 转换为 PyTorch Tensor
        # 输出形状:
        # Ego: [1, Feature_Dim]
        # Agents: [N, Feature_Dim] (N是其他单位数量)

        tensor_ego = torch.tensor([ego_token], dtype=torch.float32)

        if agent_tokens_list:
            tensor_agents = torch.tensor(agent_tokens_list, dtype=torch.float32)
        else:
            tensor_agents = torch.empty((0, self.feature_dim))

        return {
            "ego_token": tensor_ego,
            "agent_tokens": tensor_agents,
            "agent_ids": agent_ids
        }


# --- 模拟读取您上传的文件 ---
# 由于我无法直接解压您上传的二进制文本块，这里模拟一个ACMI的文本流
# 假设 Ego ID 是 "101" (通常是十六进制，如 "401")

def process_uploaded_acmi(file_path_or_content, ego_hex_id="401"):
    tokenizer = AirCombatTokenizer(ego_id=ego_hex_id)

    print(f"开始处理 ACMI 数据，以单位 ID [{ego_hex_id}] 为中心...")

    # 尝试作为Zip打开
    try:
        # 如果是真实文件路径
        if isinstance(file_path_or_content, str):
            zf = zipfile.ZipFile(file_path_or_content, 'r')
        else:
            # 如果是内存中的字节流
            zf = zipfile.ZipFile(io.BytesIO(file_path_or_content), 'r')

        # 找到里面的 .txt.acmi 文件
        file_list = zf.namelist()
        target_file = [f for f in file_list if f.endswith('.txt.acmi') or f.endswith('.acmi')][0]

        print(f"正在读取内部文件: {target_file}")

        with zf.open(target_file) as f:
            # ACMI 是流式数据，我们需要按时间帧(frame)分段
            current_frame_lines = []

            for line_bytes in f:
                line = line_bytes.decode('utf-8', errors='ignore').strip()

                # ACMI 时间戳行以 '#' 开头，例如 #0.00, #1.00
                if line.startswith('#'):
                    if current_frame_lines:
                        # 处理上一帧
                        result = tokenizer.tokenize_frame(current_frame_lines)
                        if result:
                            # --- 这里就是 GenAD 的输入数据 ---
                            print(
                                f"Frame Processed. Ego Tensor: {result['ego_token'].shape}, Agents: {result['agent_tokens'].shape}")
                            # 在这里，您可以将 result 喂给您的 LLM 或 Transformer 模型

                    # 重置下一帧
                    current_frame_lines = []
                else:
                    current_frame_lines.append(line)

    except zipfile.BadZipFile:
        print("错误：文件不是有效的Zip格式。如果已经是解压的txt，请直接读取。")
    except Exception as e:
        print(f"处理出错: {e}")

# 提示：由于我是在沙箱环境中，我无法直接读取您上传的那个二进制流。
# 您可以将上面的代码复制到本地，并将文件路径传给 process_uploaded_acmi 函数。
# 例如: process_uploaded_acmi("51st-DDCS-Round_1.acmi", ego_hex_id="1001")
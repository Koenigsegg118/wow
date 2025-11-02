import re
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Position:
    """位置和姿态数据"""
    timestamp: float
    lon: float
    lat: float
    alt: float
    roll: float = 0.0
    pitch: float = 0.0
    heading: float = 0.0


@dataclass
class TurnEvent:
    """转向事件"""
    aircraft_id: str
    aircraft_name: str
    start_time: float
    end_time: float
    start_heading: float
    end_heading: float
    turn_angle: float
    start_pos: Position
    end_pos: Position
    enemy_positions: List[Tuple[str, Position]]  # 敌方在转向时的位置


class ACMIAnalyzer:
    def __init__(self, filename: str):
        self.filename = filename
        self.objects = {}  # 对象信息
        self.trajectories = defaultdict(list)  # 轨迹数据
        self.reference_lon = 0.0
        self.reference_lat = 0.0

    def parse_acmi(self):
        """解析ACMI文件"""
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 解析参考点
                if line.startswith('ReferenceLatitude='):
                    self.reference_lat = float(line.split('=')[1])
                elif line.startswith('ReferenceLongitude='):
                    self.reference_lon = float(line.split('=')[1])

                # 解析对象定义
                elif line.startswith('#'):
                    self._parse_object(line)

                # 解析时间帧数据
                elif line.startswith('#') == False and ',' in line:
                    self._parse_frame(line)

    def _parse_object(self, line: str):
        """解析对象定义行"""
        parts = line[1:].split(',')
        obj_id = parts[0]
        obj_info = {}

        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                obj_info[key] = value

        self.objects[obj_id] = obj_info

    def _parse_frame(self, line: str):
        """解析时间帧数据"""
        parts = line.split(',')
        obj_id = parts[0]
        timestamp = float(parts[0]) if '.' in parts[0] else None

        # 如果第一个是时间戳
        if timestamp is not None:
            return

        # 解析位置数据
        pos_data = {}
        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                pos_data[key] = value
            elif part.startswith('T='):
                pos_data['T'] = part[2:]

        if 'T' in pos_data:
            # 解析位置信息 T=经度|纬度|高度|横滚|俯仰|偏航
            coords = pos_data['T'].split('|')
            if len(coords) >= 3:
                pos = Position(
                    timestamp=0.0,  # 需要从上下文获取
                    lon=float(coords[0]),
                    lat=float(coords[1]),
                    alt=float(coords[2]),
                    roll=float(coords[3]) if len(coords) > 3 else 0.0,
                    pitch=float(coords[4]) if len(coords) > 4 else 0.0,
                    heading=float(coords[5]) if len(coords) > 5 else 0.0
                )
                self.trajectories[obj_id].append(pos)

    def calculate_heading_change(self, h1: float, h2: float) -> float:
        """计算航向角变化（考虑360度循环）"""
        diff = h2 - h1
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        return diff

    def detect_sharp_turns(self, threshold: float = 30.0, time_window: float = 5.0) -> List[TurnEvent]:
        """
        检测大角度转向
        threshold: 转向角度阈值（度）
        time_window: 时间窗口（秒）
        """
        turn_events = []

        for obj_id, trajectory in self.trajectories.items():
            if len(trajectory) < 10:
                continue

            obj_info = self.objects.get(obj_id, {})
            obj_name = obj_info.get('Name', f'Unknown-{obj_id}')

            # 遍历轨迹检测转向
            for i in range(len(trajectory) - 5):
                start_pos = trajectory[i]

                # 在时间窗口内查找终点
                for j in range(i + 5, min(i + 20, len(trajectory))):
                    end_pos = trajectory[j]

                    # 计算航向变化
                    heading_change = abs(self.calculate_heading_change(
                        start_pos.heading, end_pos.heading
                    ))

                    if heading_change >= threshold:
                        # 找到转向事件，获取敌方位置
                        enemy_positions = self._get_enemy_positions(obj_id, i)

                        turn_event = TurnEvent(
                            aircraft_id=obj_id,
                            aircraft_name=obj_name,
                            start_time=i,
                            end_time=j,
                            start_heading=start_pos.heading,
                            end_heading=end_pos.heading,
                            turn_angle=heading_change,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            enemy_positions=enemy_positions
                        )
                        turn_events.append(turn_event)
                        break

        return turn_events

    def _get_enemy_positions(self, obj_id: str, time_index: int) -> List[Tuple[str, Position]]:
        """获取转向时刻敌方飞机的位置"""
        my_coalition = self.objects.get(obj_id, {}).get('Coalition', 'Unknown')
        enemy_positions = []

        for other_id, trajectory in self.trajectories.items():
            if other_id == obj_id:
                continue

            other_coalition = self.objects.get(other_id, {}).get('Coalition', 'Unknown')

            # 判断是否为敌方
            if my_coalition != other_coalition and other_coalition != 'Unknown':
                if time_index < len(trajectory):
                    other_name = self.objects.get(other_id, {}).get('Name', f'Unknown-{other_id}')
                    enemy_positions.append((other_name, trajectory[time_index]))

        return enemy_positions

    def calculate_distance(self, pos1: Position, pos2: Position) -> float:
        """计算两点之间的距离（简化的球面距离，单位：公里）"""
        R = 6371  # 地球半径

        lat1, lon1 = math.radians(pos1.lat), math.radians(pos1.lon)
        lat2, lon2 = math.radians(pos2.lat), math.radians(pos2.lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        distance = R * c

        # 考虑高度差
        alt_diff = (pos2.alt - pos1.alt) / 1000  # 转换为公里
        distance = math.sqrt(distance ** 2 + alt_diff ** 2)

        return distance

    def print_turn_analysis(self, turn_events: List[TurnEvent]):
        """打印转向分析结果"""
        print(f"\n{'=' * 80}")
        print(f"检测到 {len(turn_events)} 次大角度转向事件")
        print(f"{'=' * 80}\n")

        for idx, event in enumerate(turn_events, 1):
            print(f"转向事件 #{idx}")
            print(f"  飞机: {event.aircraft_name} (ID: {event.aircraft_id})")
            print(f"  转向角度: {event.turn_angle:.1f}°")
            print(f"  航向变化: {event.start_heading:.1f}° → {event.end_heading:.1f}°")
            print(f"\n  转向开始位置:")
            print(f"    经度: {event.start_pos.lon:.6f}°")
            print(f"    纬度: {event.start_pos.lat:.6f}°")
            print(f"    高度: {event.start_pos.alt:.0f}m")
            print(f"\n  转向结束位置:")
            print(f"    经度: {event.end_pos.lon:.6f}°")
            print(f"    纬度: {event.end_pos.lat:.6f}°")
            print(f"    高度: {event.end_pos.alt:.0f}m")

            # 分析敌方位置
            if event.enemy_positions:
                print(f"\n  转向时敌方位置:")
                for enemy_name, enemy_pos in event.enemy_positions:
                    distance = self.calculate_distance(event.start_pos, enemy_pos)
                    print(f"    {enemy_name}:")
                    print(f"      距离: {distance:.2f} km")
                    print(f"      位置: ({enemy_pos.lon:.6f}°, {enemy_pos.lat:.6f}°)")
                    print(f"      高度: {enemy_pos.alt:.0f}m")
                    print(f"      航向: {enemy_pos.heading:.1f}°")
            else:
                print(f"\n  未检测到敌方飞机")

            print(f"\n{'-' * 80}\n")


# 使用示例
if __name__ == "__main__":
    # 使用方法
    print("ACMI飞行记录大角度转向分析工具")
    print("=" * 80)

    # 替换为你的ACMI文件路径
    acmi_file = "flight_record.acmi"

    try:
        analyzer = ACMIAnalyzer(acmi_file)
        print(f"正在解析文件: {acmi_file}")
        analyzer.parse_acmi()

        print(f"共解析 {len(analyzer.objects)} 个对象")
        print(f"共记录 {sum(len(t) for t in analyzer.trajectories.values())} 个轨迹点")

        # 检测大角度转向（默认30度以上）
        turn_events = analyzer.detect_sharp_turns(threshold=30.0)

        # 打印分析结果
        analyzer.print_turn_analysis(turn_events)

    except FileNotFoundError:
        print(f"\n错误: 找不到文件 '{acmi_file}'")
        print("请将ACMI文件放在同一目录下，或修改文件路径")
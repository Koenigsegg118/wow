import xml.etree.ElementTree as ET
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import csv


@dataclass
class Position:
    """位置数据"""
    timestamp: float
    lon: float
    lat: float
    alt: float


@dataclass
class Aircraft:
    """飞机信息"""
    id: str
    name: str
    pilot: str
    coalition: str
    country: str
    group: str


@dataclass
class MissileLaunchEvent:
    """导弹发射事件"""
    time: float
    shooter: Aircraft
    shooter_pos: Position
    weapon_name: str
    weapon_id: str
    target_info: Optional[str] = None
    distance_to_nearest_enemy: float = 0.0
    nearest_enemy: Optional[Aircraft] = None
    nearest_enemy_pos: Optional[Position] = None


@dataclass
class StateSnapshot:
    """某一时刻的状态快照"""
    time: float
    aircraft: Aircraft
    position: Position
    enemies: List[Tuple[Aircraft, Position, float, float]] = field(default_factory=list)


class ACMIMissileAnalyzer:
    def __init__(self, filename: str):
        self.filename = filename
        self.tree = None
        self.root = None
        self.aircrafts = {}  # ID -> Aircraft
        self.all_events = []  # 所有事件的时间序列
        self.missile_launches = []  # List[MissileLaunchEvent]

    def parse_xml(self):
        """解析XML文件"""
        print(f"正在解析XML文件: {self.filename}")

        try:
            self.tree = ET.parse(self.filename)
            self.root = self.tree.getroot()
        except ET.ParseError as e:
            print(f"XML解析错误: {e}")
            return False

        # 解析所有事件
        events = self.root.findall('.//Event')
        print(f"找到 {len(events)} 个事件")

        for event in events:
            time_elem = event.find('Time')
            if time_elem is None:
                continue

            timestamp = float(time_elem.text)

            # 解析位置
            location = event.find('Location')
            if location is None:
                continue

            lon_elem = location.find('Longitude')
            lat_elem = location.find('Latitude')
            alt_elem = location.find('Altitude')

            if lon_elem is None or lat_elem is None or alt_elem is None:
                continue

            lon = float(lon_elem.text)
            lat = float(lat_elem.text)
            alt = float(alt_elem.text)

            # 解析主要对象
            primary_obj = event.find('PrimaryObject')
            if primary_obj is None:
                continue

            obj_id = primary_obj.get('ID')
            type_elem = primary_obj.find('Type')

            # 保存飞机信息
            if type_elem is not None and type_elem.text == 'Aircraft':
                if obj_id not in self.aircrafts:
                    aircraft = Aircraft(
                        id=obj_id,
                        name=primary_obj.find('Name').text if primary_obj.find('Name') is not None else 'Unknown',
                        pilot=primary_obj.find('Pilot').text if primary_obj.find('Pilot') is not None else 'Unknown',
                        coalition=primary_obj.find('Coalition').text if primary_obj.find(
                            'Coalition') is not None else 'Unknown',
                        country=primary_obj.find('Country').text if primary_obj.find(
                            'Country') is not None else 'Unknown',
                        group=primary_obj.find('Group').text if primary_obj.find('Group') is not None else 'Unknown'
                    )
                    self.aircrafts[obj_id] = aircraft

            # 保存事件
            action_elem = event.find('Action')
            action = action_elem.text if action_elem is not None else None

            event_data = {
                'time': timestamp,
                'obj_id': obj_id,
                'type': type_elem.text if type_elem is not None else None,
                'action': action,
                'position': Position(timestamp, lon, lat, alt),
                'event': event
            }
            self.all_events.append(event_data)

            # 检测导弹发射事件
            if action == 'HasFired':
                secondary_obj = event.find('SecondaryObject')
                if secondary_obj is not None:
                    weapon_id = secondary_obj.get('ID')
                    weapon_name_elem = secondary_obj.find('Name')
                    weapon_name = weapon_name_elem.text if weapon_name_elem is not None else 'Unknown Weapon'

                    if obj_id in self.aircrafts:
                        shooter_pos = Position(timestamp, lon, lat, alt)

                        launch_event = MissileLaunchEvent(
                            time=timestamp,
                            shooter=self.aircrafts[obj_id],
                            shooter_pos=shooter_pos,
                            weapon_name=weapon_name,
                            weapon_id=weapon_id
                        )
                        self.missile_launches.append(launch_event)

        # 按时间排序事件
        self.all_events.sort(key=lambda x: x['time'])

        # 为每次发射找到最近的敌机
        for launch in self.missile_launches:
            self._find_nearest_enemy_at_launch(launch)

        print(f"\n解析完成:")
        print(f"  - 飞机数量: {len(self.aircrafts)}")
        print(f"  - 总事件数: {len(self.all_events)}")
        print(f"  - 导弹发射次数: {len(self.missile_launches)}")

        return True

    def _find_nearest_enemy_at_launch(self, launch: MissileLaunchEvent):
        """找到发射时刻最近的敌机"""
        launch_time = launch.time

        # 找到发射时刻前后5秒内的所有敌方飞机位置
        min_distance = float('inf')
        nearest_enemy = None
        nearest_pos = None

        for event_data in self.all_events:
            # 只看发射时刻前后5秒
            if abs(event_data['time'] - launch_time) > 5.0:
                continue

            # 只看飞机类型的事件
            if event_data['type'] != 'Aircraft':
                continue

            other_id = event_data['obj_id']
            if other_id == launch.shooter.id:
                continue

            if other_id not in self.aircrafts:
                continue

            other_aircraft = self.aircrafts[other_id]

            # 检查是否为敌方
            if launch.shooter.coalition != other_aircraft.coalition:
                distance = self.calculate_distance(launch.shooter_pos, event_data['position'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_enemy = other_aircraft
                    nearest_pos = event_data['position']

        if nearest_enemy:
            launch.nearest_enemy = nearest_enemy
            launch.nearest_enemy_pos = nearest_pos
            launch.distance_to_nearest_enemy = min_distance

    def get_states_around_launch(self, launch: MissileLaunchEvent, time_window: float = 5.0) -> List[StateSnapshot]:
        """获取发射前后时间窗口内的状态快照"""
        snapshots = []

        start_time = launch.time - time_window
        end_time = launch.time + time_window

        # 收集发射者在这个时间窗口内的所有位置
        shooter_events = defaultdict(lambda: None)

        for event_data in self.all_events:
            if event_data['obj_id'] == launch.shooter.id and start_time <= event_data['time'] <= end_time:
                if event_data['type'] == 'Aircraft':
                    shooter_events[event_data['time']] = event_data['position']

        # 为每个时间点创建快照
        for time_point in sorted(shooter_events.keys()):
            pos = shooter_events[time_point]

            # 找到这个时刻所有敌机的位置
            enemies = []
            for event_data in self.all_events:
                if abs(event_data['time'] - time_point) > 2.0:  # 2秒容差
                    continue

                if event_data['type'] != 'Aircraft':
                    continue

                other_id = event_data['obj_id']
                if other_id == launch.shooter.id or other_id not in self.aircrafts:
                    continue

                other_aircraft = self.aircrafts[other_id]
                if launch.shooter.coalition != other_aircraft.coalition:
                    distance = self.calculate_distance(pos, event_data['position'])
                    bearing = self.calculate_bearing(pos, event_data['position'])
                    enemies.append((other_aircraft, event_data['position'], distance, bearing))

            # 去重：同一飞机只保留最近的一个记录
            unique_enemies = {}
            for enemy_aircraft, enemy_pos, distance, bearing in enemies:
                if enemy_aircraft.id not in unique_enemies or distance < unique_enemies[enemy_aircraft.id][2]:
                    unique_enemies[enemy_aircraft.id] = (enemy_aircraft, enemy_pos, distance, bearing)

            snapshot = StateSnapshot(
                time=time_point,
                aircraft=launch.shooter,
                position=pos,
                enemies=list(unique_enemies.values())
            )
            snapshots.append(snapshot)

        return snapshots

    def calculate_distance(self, pos1: Position, pos2: Position) -> float:
        """计算两点之间的距离（单位：公里）"""
        R = 6371

        lat1, lon1 = math.radians(pos1.lat), math.radians(pos1.lon)
        lat2, lon2 = math.radians(pos2.lat), math.radians(pos2.lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        distance = R * c

        alt_diff = (pos2.alt - pos1.alt) / 1000
        distance = math.sqrt(distance ** 2 + alt_diff ** 2)

        return distance

    def calculate_bearing(self, pos1: Position, pos2: Position) -> float:
        """计算方位角"""
        lat1, lon1 = math.radians(pos1.lat), math.radians(pos1.lon)
        lat2, lon2 = math.radians(pos2.lat), math.radians(pos2.lon)

        dlon = lon2 - lon1

        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360

        return bearing

    def export_to_csv(self, output_file: str = "missile_launches_analysis.csv"):
        """导出导弹发射分析到CSV文件"""
        print(f"\n正在导出CSV文件: {output_file}")

        with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)

            # 写入表头
            header = [
                '发射序号', '发射时间(s)', '发射者', '机型', '阵营', '编队', '武器',
                '发射者经度', '发射者纬度', '发射者高度(m)', '发射者高度(ft)',
                '最近敌机', '敌机机型', '敌机阵营',
                '距离(km)', '距离(NM)',
                '敌机经度', '敌机纬度', '敌机高度(m)', '敌机高度(ft)',
                '高度差(m)', '高度差(ft)', '方位角(deg)'
            ]
            writer.writerow(header)

            # 写入每次发射的数据
            for idx, launch in enumerate(self.missile_launches, 1):
                row = [
                    idx,
                    f"{launch.time:.2f}",
                    launch.shooter.pilot,
                    launch.shooter.name,
                    launch.shooter.coalition,
                    launch.shooter.group,
                    launch.weapon_name,
                    f"{launch.shooter_pos.lon:.6f}",
                    f"{launch.shooter_pos.lat:.6f}",
                    f"{launch.shooter_pos.alt:.1f}",
                    f"{launch.shooter_pos.alt / 0.3048:.0f}",
                ]

                if launch.nearest_enemy and launch.nearest_enemy_pos:
                    distance_km = launch.distance_to_nearest_enemy
                    distance_nm = distance_km * 0.539957
                    alt_diff = launch.nearest_enemy_pos.alt - launch.shooter_pos.alt
                    bearing = self.calculate_bearing(launch.shooter_pos, launch.nearest_enemy_pos)

                    row.extend([
                        launch.nearest_enemy.pilot,
                        launch.nearest_enemy.name,
                        launch.nearest_enemy.coalition,
                        f"{distance_km:.2f}",
                        f"{distance_nm:.2f}",
                        f"{launch.nearest_enemy_pos.lon:.6f}",
                        f"{launch.nearest_enemy_pos.lat:.6f}",
                        f"{launch.nearest_enemy_pos.alt:.1f}",
                        f"{launch.nearest_enemy_pos.alt / 0.3048:.0f}",
                        f"{alt_diff:.1f}",
                        f"{alt_diff / 0.3048:.0f}",
                        f"{bearing:.1f}"
                    ])
                else:
                    row.extend(['', '', '', '', '', '', '', '', '', '', '', ''])

                writer.writerow(row)

        print(f"✓ CSV文件已保存")

    def export_time_series_csv(self, output_file: str = "missile_launch_timeseries.csv", time_window: float = 5.0):
        """导出每次发射前后5秒的时间序列数据"""
        print(f"\n正在导出时间序列CSV文件: {output_file}")

        with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)

            # 写入表头
            header = [
                '发射序号', '发射者', '武器', '最近目标',
                '相对时间(s)', '绝对时间(s)',
                '发射者经度', '发射者纬度', '发射者高度(m)', '发射者高度(ft)',
                '敌机数量', '最近敌机', '最近敌机距离(km)', '最近敌机距离(NM)', '最近敌机方位(deg)',
                '最近敌机经度', '最近敌机纬度', '最近敌机高度(m)', '最近敌机高度(ft)',
                '高度差(m)', '高度差(ft)'
            ]
            writer.writerow(header)

            # 为每次发射生成时间序列
            for idx, launch in enumerate(self.missile_launches, 1):
                target_name = launch.nearest_enemy.pilot if launch.nearest_enemy else 'Unknown'
                snapshots = self.get_states_around_launch(launch, time_window)

                for snapshot in snapshots:
                    relative_time = snapshot.time - launch.time

                    # 找到最近的敌机
                    closest_enemy = None
                    closest_distance = float('inf')

                    for enemy_aircraft, enemy_pos, distance, bearing in snapshot.enemies:
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_enemy = (enemy_aircraft, enemy_pos, bearing)

                    row = [
                        idx,
                        launch.shooter.pilot,
                        launch.weapon_name,
                        target_name,
                        f"{relative_time:.2f}",
                        f"{snapshot.time:.2f}",
                        f"{snapshot.position.lon:.6f}",
                        f"{snapshot.position.lat:.6f}",
                        f"{snapshot.position.alt:.1f}",
                        f"{snapshot.position.alt / 0.3048:.0f}",
                        len(snapshot.enemies)
                    ]

                    if closest_enemy:
                        enemy_aircraft, enemy_pos, bearing = closest_enemy
                        alt_diff = enemy_pos.alt - snapshot.position.alt
                        row.extend([
                            enemy_aircraft.pilot,
                            f"{closest_distance:.2f}",
                            f"{closest_distance * 0.539957:.2f}",
                            f"{bearing:.1f}",
                            f"{enemy_pos.lon:.6f}",
                            f"{enemy_pos.lat:.6f}",
                            f"{enemy_pos.alt:.1f}",
                            f"{enemy_pos.alt / 0.3048:.0f}",
                            f"{alt_diff:.1f}",
                            f"{alt_diff / 0.3048:.0f}"
                        ])
                    else:
                        row.extend(['', '', '', '', '', '', '', '', '', ''])

                    writer.writerow(row)

                # 添加空行分隔不同的发射事件
                writer.writerow([])

        print(f"✓ 时间序列CSV文件已保存")

    def print_launch_summary(self):
        """打印导弹发射总览"""
        print(f"\n{'=' * 100}")
        print(f"导弹发射事件总览")
        print(f"{'=' * 100}\n")

        # 按阵营统计
        allies_launches = [l for l in self.missile_launches if l.shooter.coalition == 'Allies']
        enemies_launches = [l for l in self.missile_launches if l.shooter.coalition == 'Enemies']

        print(f"总发射次数: {len(self.missile_launches)}")
        print(f"  - 红方 (Allies): {len(allies_launches)} 次")
        print(f"  - 蓝方 (Enemies): {len(enemies_launches)} 次")
        print()

        for idx, launch in enumerate(self.missile_launches, 1):
            print(f"【发射 #{idx}】")
            print(f"  时间: {launch.time:.2f}s")
            print(f"  发射者: {launch.shooter.pilot} ({launch.shooter.name})")
            print(f"  阵营: {launch.shooter.coalition}")
            print(f"  编队: {launch.shooter.group}")
            print(f"  武器: {launch.weapon_name}")
            print(f"  位置: ({launch.shooter_pos.lon:.4f}°, {launch.shooter_pos.lat:.4f}°)")
            print(f"  高度: {launch.shooter_pos.alt:.0f}m ({launch.shooter_pos.alt / 0.3048:.0f}ft)")

            if launch.nearest_enemy and launch.nearest_enemy_pos:
                print(f"\n  最近敌机: {launch.nearest_enemy.pilot} ({launch.nearest_enemy.name})")
                print(
                    f"  距离: {launch.distance_to_nearest_enemy:.2f} km ({launch.distance_to_nearest_enemy * 0.539957:.2f} NM)")

                alt_diff = launch.nearest_enemy_pos.alt - launch.shooter_pos.alt
                print(
                    f"  敌机高度: {launch.nearest_enemy_pos.alt:.0f}m ({launch.nearest_enemy_pos.alt / 0.3048:.0f}ft)")
                print(f"  高度差: {alt_diff:+.0f}m ({alt_diff / 0.3048:+.0f}ft)")

                bearing = self.calculate_bearing(launch.shooter_pos, launch.nearest_enemy_pos)
                print(f"  方位角: {bearing:.1f}°")
            else:
                print(f"\n  ⚠️ 未找到敌机（雷达范围外或数据不足）")

            print()


# 使用示例
if __name__ == "__main__":
    print("=" * 100)
    print("ACMI导弹发射态势分析工具")
    print("=" * 100)
    print()

    # XML文件路径
    xml_file = "104th vs inSky - Rd 1.xml"

    try:
        analyzer = ACMIMissileAnalyzer(xml_file)

        if analyzer.parse_xml():
            # 打印发射总览
            analyzer.print_launch_summary()

            # 导出发射时刻的详细数据
            analyzer.export_to_csv("missile_launches_analysis.csv")

            # 导出发射前后5秒的时间序列数据
            analyzer.export_time_series_csv("missile_launch_timeseries.csv", time_window=5.0)

            print("\n✓ 分析完成！")
            print(f"  - 发射总览: missile_launches_analysis.csv")
            print(f"  - 时间序列: missile_launch_timeseries.csv")
        else:
            print("文件解析失败")

    except FileNotFoundError:
        print(f"\n❌ 错误: 找不到文件 '{xml_file}'")
        print("请确保XML文件在同一目录下，或修改代码中的文件路径")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback

        traceback.print_exc()
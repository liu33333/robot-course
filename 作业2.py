import math
import struct
from dataclasses import dataclass

import matplotlib.axes
import matplotlib.pyplot as plt


@dataclass
class Bytes:
    '''不同数据字节数'''

    float = 4
    long = 4
    unsigned_short = 2
    header: int  # LMS头部
    lmsdatbuf: int  # 单条LMS数据


@dataclass
class NAVData:
    '''单条NAV数据'''

    milli: int
    yaw: float
    x: float
    y: float


class LMSDATBUF:
    '''单条LMS数据'''

    Unit: float
    AngRes: float

    def __init__(self, milli: int, dat: list[int]):
        self.milli = milli
        self.dat = dat

    def cal_world_position(self, nav: NAVData) -> list[tuple[float, float]]:
        '''计算激光点全局坐标

        Args:
            nav (NAVData): NAV数据

        Returns:
            list[tuple[float,float]]: 所有激光点坐标
        '''

        self.world_p = []
        for i in range(len(self.dat)):
            r = self.dat[i] / self.Unit  # 激光距离
            world_a = nav.yaw + i * self.AngRes  # 激光全局角度 = 机器人角度 + 激光扫描角
            # 局部坐标 = 机器人坐标 + 激光距离 * 激光全局角度
            delta_x = r * math.cos(world_a)
            delta_y = r * math.sin(world_a)
            self.world_p.append((nav.x + delta_x, nav.y + delta_y))

        return self.world_p


class GridMap:
    '''栅格地图'''

    length = 0.1  # 栅格边长
    threshold = 16  # 简易投票阈值

    def __init__(self, ps: list[list[float, float]]):
        '''
        Args:
            ps (list[list[float, float]]): 所有激光点坐标
        '''

        self.ps = ps
        self.xs, self.ys = zip(*self.ps)

        # 栅格地图边界
        self.left = math.floor(min(self.xs))
        self.bottom = math.floor(min(self.ys))
        right = math.ceil(max(self.xs))
        top = math.ceil(max(self.ys))

        # 栅格地图尺寸
        self.width = math.ceil((right - self.left) / self.length)
        self.height = math.ceil((top - self.bottom) / self.length)

    def vote(self) -> None:
        '''简易投票法'''

        # 栅格地图
        self.map = [[0] * self.height for _ in range(self.width)]
        for x, y in self.ps:
            # 坐标 -> 栅格地图下标
            i = math.floor((x - self.left) / self.length)
            j = math.floor((y - self.bottom) / self.length)
            # 投票
            self.map[i][j] += 1

        # 投票出来的栅格坐标
        self.voted = []
        for i in range(self.width):
            for j in range(self.height):
                if self.map[i][j] >= self.threshold:
                    # 栅格地图下标 -> 坐标
                    x = self.left + i * self.length
                    y = self.bottom + j * self.length
                    self.voted.append((x, y))

    def show(self) -> None:
        '''显示栅格地图'''

        ax: matplotlib.axes._axes.Axes
        fig, ax = plt.subplots()

        # 显示所有激光点
        # ax.scatter(self.xs, self.ys, s=0.1, c='red', linewidths=0)

        # 显示投票出来的栅格
        for xy in self.voted:
            ax.add_patch(plt.Rectangle(xy, self.length, self.length))

        plt.axis('equal')
        plt.show()


def read_lms_data(file_path: str) -> list[LMSDATBUF]:
    '''读取LMS数据

    Args:
        file_path (str): LMS文件路径

    Returns:
        list[LMSDATBUF]: LMS数据
    '''

    with open(file_path, 'rb') as f:
        urg = f.read()

    len_header = 3  # LMS头部数据数
    Bytes.header = len_header * Bytes.float  # LMS头部数据字节数
    urg_header = urg[: Bytes.header]  # LMS头部数据
    urg_lmsdatbuf = urg[Bytes.header :]  # LMS数据

    AngRng, AngRes, Unit = struct.unpack(f'<{len_header}f', urg_header)  # 解析头部数据
    DATLEN = int(AngRng / AngRes + 1)  # 激光扫描点数
    Bytes.lmsdatbuf = Bytes.long + DATLEN * Bytes.unsigned_short
    LMSDATBUF.Unit = Unit
    LMSDATBUF.AngRes = math.radians(AngRes)

    lms_data = []
    # 解析每条完整的LMS数据
    for i in range(0, len(urg_lmsdatbuf) // Bytes.lmsdatbuf * Bytes.lmsdatbuf, Bytes.lmsdatbuf):
        milli, *dat = struct.unpack(f'<l{DATLEN}H', urg_lmsdatbuf[i : i + Bytes.lmsdatbuf])
        lms_data.append(LMSDATBUF(milli, dat))

    return lms_data


def read_nav_data(file_path: str) -> list[NAVData]:
    '''读取NAV数据

    Args:
        file_path (str): NAV文件路径

    Returns:
        list[NAVData]: NAV数据
    '''

    with open(file_path) as f:
        nav_data = f.read()

    nav_data = nav_data.splitlines()[1:]  # 按行分割，跳过第一行
    nav_data = [line.split() for line in nav_data]  # 分割每行
    nav_data = [NAVData(int(line[0]), *map(float, line[3:6])) for line in nav_data]  # 解析每行

    return nav_data


def filter_data(
    lms_data: list[LMSDATBUF], nav_data: list[NAVData]
) -> tuple[list[LMSDATBUF], list[NAVData]]:
    '''过滤出有效时间戳的数据

    Args:
        lms_data (list[LMSDATBUF]): LMS数据
        nav_data (list[NAVData]):NAV数据

    Returns:
        tuple[list[LMSDATBUF], list[NAVData]]: LMS数据和NAV数据
    '''

    start_milli = 71439698  # 起始时间戳为机器人开始移动的时间
    end_milli = nav_data[-1].milli  # 结束时间戳为最后一条NAV数据的时间戳

    lms_data = list(filter(lambda x: start_milli <= x.milli <= end_milli, lms_data))
    nav_data = list(filter(lambda x: x.milli >= start_milli, nav_data))

    assert len(lms_data) == len(nav_data), 'LMS和NAV数据长度不一致'
    assert all(
        lms.milli == nav.milli for lms, nav in zip(lms_data, nav_data)
    ), 'LMS和NAV数据时间戳不一致'

    return lms_data, nav_data


def main():
    lms_data = read_lms_data('URG_X_20130903_195003.lms')
    nav_data = read_nav_data('ld.nav')
    lms_data, nav_data = filter_data(lms_data, nav_data)

    ps = []  # 所有激光点坐标
    for lms, nav in zip(lms_data, nav_data):
        ps.extend(lms.cal_world_position(nav))

    grid_map = GridMap(ps)
    grid_map.vote()
    grid_map.show()


if __name__ == '__main__':
    main()

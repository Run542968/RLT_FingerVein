import math
import random
import numpy as np
import cv2


class RLT():
    def __init__(self, p_lr, p_ud, W, r, N, threshold, multiple):
        self.p_lr = p_lr
        self.p_ud = p_ud
        self.W = W
        self.W_r = math.ceil(W / 2)
        self.r = r
        self.N = N  # 迭代次数
        self.threshold = threshold
        self.multiple = multiple

    # 构造Rnd函数——返回一个均匀随机数
    def Rnd(self, n):
        return random.uniform(0, n)

    # 选择开始点——返回在手指轮廓中的开始点Xs,Ys以及Dlr和Dud
    def Choose_Start(self, img, row, col):
        Xs = random.randint(self.W_r, col - self.W_r - 1)  ## 取开始点的时候要考虑别取到边界值,限定范围在[2/W,col-2/W-1]
        Ys = random.randint(self.W_r, row - self.W_r - 1)
        while (img[Ys][Xs] == 0):  ##注意：坐标（x，y）对应的是（列，行）
            Xs = random.randint(self.W_r, col - self.W_r - 1)
            Ys = random.randint(self.W_r, row - self.W_r - 1)

        if (self.Rnd(2) < 1):
            D_lr = (1, 0)
            D_ud = (0, 1)
        else:
            D_lr = (-1, 0)
            D_ud = (0, -1)

        return Xs, Ys, D_lr, D_ud

    # 得到当前点可以移动的Nc的集合
    def get_Nr(self, curr_point, p_lr, p_ud, D_lr, D_ud):
        Xc = curr_point[0]
        Yc = curr_point[1]
        t = self.Rnd(100)  ##定义一个局部变量，只调用一个Rnd(n)，避免每次if的时候都重新调用Rnd(n)函数
        if (t < p_lr):
            return [(D_lr[0] + Xc, D_lr[1] + Yc), (D_lr[0] - D_lr[1] + Xc, D_lr[1] - D_lr[0] + Yc),
                    (D_lr[0] + D_lr[1] + Xc, D_lr[1] + D_lr[0] + Yc)]
        elif (t >= p_lr and t < p_lr + p_ud):
            return [(D_ud[0] + Xc, D_ud[1] + Yc), (D_ud[0] - D_ud[1] + Xc, D_ud[1] - D_ud[0] + Yc),
                    (D_ud[0] + D_ud[1] + Xc, D_ud[1] + D_ud[0] + Yc)]
        elif (t >= p_lr + p_ud):
            return [(Xc - 1, Yc), (Xc - 1, Yc + 1), (Xc, Yc + 1), (Xc + 1, Yc + 1), (Xc + 1, Yc), (Xc + 1, Yc - 1),
                    (Xc, Yc - 1), (Xc - 1, Yc - 1)]

    def compute_angle(self, vector1, vector2):  # 返回弧度制的seta角度
        u = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        d = math.sqrt(math.pow(vector1[0], 2) + math.pow(vector1[1], 2)) * math.sqrt(
            math.pow(vector2[0], 2) + math.pow(vector2[1], 2))
        cos = u / d
        seta = math.acos(cos)
        if vector2[1] >= 0:
            return seta
        else:
            return -seta

    def extract(self, img, row, col, right_limit, left_limit, up_limit, down_limit, locus_space):
        for count in range(self.N):
            # 选择开始点
            Xs, Ys, D_lr, D_ud = self.Choose_Start(img, row, col)
            Xc, Yc = Xs, Ys

            # 检测黑线的方向和追踪点的移动
            # 初始化Tc
            Tc = []
            while True:
                # 决定Nc——Nc中的每个坐标都是Xi,Yi
                Nc = []
                # print("Xc,Yc={0}".format((Xc, Yc)))
                Nr = self.get_Nr((Xc, Yc), self.p_lr, self.p_ud, D_lr, D_ud)  # 计算出Nr
                for i in Nr:  # Nr∩Rf
                    if (img[i[1]][i[0]] > 0):
                        Nc.append(i)
                Nc = list(set(Nc).difference(set(Tc)))  # Nr∩Rf∩(~Tc)
                # 求出Vl
                max_Vl = float("-inf")
                max_point = ()
                for j in Nc:
                    vector1 = (1, 0)
                    vector2 = (j[0] - Xc, j[1] - Yc)
                    seta = self.compute_angle(vector1, vector2)
                    # print('vector2:{0},seta:{1}'.format(vector2,seta))
                    c = math.cos(seta)
                    s = math.sin(seta)
                    Vl = int(img[round(Yc + self.r * s + (self.W / 2) * c)][
                                 round(Xc + self.r * c - (self.W / 2) * s)]) + int(
                        img[round(Yc + self.r * s - (self.W / 2) * c)][round(Xc + self.r * c + (self.W / 2) * s)]) \
                         - 2 * int(img[round(Yc + self.r * s)][round(Xc + self.r * c)])
                    # print('当j为{}时，三个关键点分别为：{}，{}，{},Vl为{}'.format(j,(round(Xc+r*math.cos(seta)-(W/2)*math.sin(seta)),round(Yc+r*math.sin(seta)+(W/2)*math.cos(seta))),\
                    #                                            (round(Xc+r*math.cos(seta)+(W/2)*math.sin(seta)),round(Yc+r*math.sin(seta)-(W/2)*math.cos(seta))),\
                    #                                            (round(Xc+r*math.cos(seta)),round(Yc+r*math.sin(seta))),Vl ))
                    if (Vl >= max_Vl):
                        max_Vl = Vl
                        max_point = j
                # 更新当前tracking point
                Tc.append((Xc, Yc))
                if (max_Vl > 0 and max_point[0] <= right_limit and max_point[0] >= left_limit and max_point[
                    1] <= up_limit and max_point[1] >= down_limit):
                    Xc, Yc = max_point[0], max_point[1]
                else:
                    break
            # print('Tc:{}'.format(Tc))

            # 更新locus space中的点信息
            for k in Tc:
                locus_space[k[1]][k[0]] = locus_space[k[1]][k[0]] + 1
            print("当前为第%d次迭代" % (count))

        return locus_space

    def fliter(self, locus_space):
        locus_space *= self.multiple
        for m in range(locus_space.shape[0]):
            for n in range(locus_space.shape[1]):
                if (locus_space[m][n] > self.threshold):
                    locus_space[m][n] = 255
                else:
                    locus_space[m][n] = 0
        return locus_space

    def dir_limit(self,row,col):
        up_limit = row - self.W_r - 1
        down_limit = self.W_r
        right_limit = col - self.W_r - 1
        left_limit = self.W_r
        return up_limit,down_limit,right_limit,left_limit

if __name__ == "__main__":
    img = cv2.imread('../data/origin/02.bmp', 2)  # 读取模式为灰度图
    row,col=img.shape[0],img.shape[1]
    print("图像的大小%d x %d" % (img.shape[0], img.shape[1]))
    # 新建一个Tr(x,y)——locus space
    locus_space = np.zeros((row, col), dtype=np.uint8)

    rlt=RLT(50,25,W=11,r=1,N=500,threshold=70,multiple=40)#这个参数挺适合128，60的MUM6000_ROIs原图
    up_limit, down_limit, right_limit, left_limit=rlt.dir_limit(row,col)
    locus_space=rlt.extract(img,row,col,right_limit,left_limit,up_limit,down_limit,locus_space)
    locus_space=rlt.fliter(locus_space)
    print(locus_space.shape)
    cv2.imwrite('1rois.bmp',locus_space)
    cv2.imshow('filetitle', locus_space)
    cv2.waitKey(0)


##用于提取特征

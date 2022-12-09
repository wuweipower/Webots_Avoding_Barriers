from enum import Enum
import numpy as np
import math
import inspect
import time
from controller import Robot,GPS
from controller import Camera
#from controller import CameraRecognitionObject
from controller import InertialUnit

robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor= robot.getDevice('motor_1')
right_motor= robot.getDevice('motor_2')
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

left_ps=robot.getDevice('ps_1')
left_ps.enable(timestep)
right_ps=robot.getDevice('ps_2')
right_ps.enable(timestep)

imu = robot.getDevice("inertial unit")
imu.enable(timestep)

gps=robot.getDevice('gps')
gps.enable(timestep)

camera = robot.getDevice("camera")
camera.enable(timestep)

#允许识别
camera.recognitionEnable(timestep)

# gps=robot.getDevice('gps')
# gps.enable(timestep)
class RobotType(Enum):
    circle = 0
    rectangle = 1

#将一些常量封装
class Config:
    def __init__(self):
        # 速度及加速度
        self.max_speed = 5.5
        self.min_speed = -3
        self.max_accel = 10
        self.dist_btw_wheels=9
        self.wheel_radius=2.5
        self.dist_per_radian=self.wheel_radius
        #偏航角度 角加速度
        self.max_yaw_rate = 30.0 * math.pi / 180.0  # [rad/s]
        self.max_delta_yaw_rate = 120.0 * math.pi / 180.0  # [rad/ss]
        #角速度的切片
        self.yaw_rate_resolution = 5 * math.pi / 180.0  # [rad/s]

        #速度切片
        self.v_resolution = 0.5

        #障碍物的半径
        self.obstacle_radius=10
        #时间区间及时间片长度
        self.dt = 0.2  # [s] Time tick for motion prediction
        self.predict_time =1.0 # [s]
        #评价函数的三个权值
        self.to_goal_cost_gain = 1.5
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.3

        #防止被卡住
        self.robot_stuck_flag_cons = 0.001 
        self.robot_type = RobotType.circle

        #世界的大小
        self.board_width=500
        self.board_length=500

        #用来check碰撞
        self.robot_radius =10 
        #预先设置一些全局静态障碍物的位置 这里x表示为机器人坐标z，y表示机器人坐标x
        self.ob = np.array([[-150,-160],[-125,-50],[0,0],[72,120]])
    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value

config = Config()  

#trajectory轨迹
def dwa_control(x, config, goal, obstacle):
    dw = calc_dynamic_window(x, config)#dw dynamic window
    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, obstacle)#u就是[v,w]
    return u, trajectory

#计算动态窗口
def calc_dynamic_window(x, config):
    '''
    x就是机器人的运动学描述，五个元素的矩阵[x,y,v,theta,w]
    '''
    Vs = [config.min_speed, config.max_speed,-config.max_yaw_rate, config.max_yaw_rate]#线速度和角速度的取值范围

    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    return dw

def calc_to_goal_cost(trajectory, goal):
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    return math.hypot(dx,dy)

'''
x(t+∆t)=x(t)+v(t)*cos(θ(t))*∆t
y(t+∆t)=y(t)+v(t)*sin(θ(t))*∆t
θ(t+∆t)=θ(t)+w(t)*∆t
v(t+∆t)=v(t)+a(t)*∆t
ω(t+∆t)=ω(t)+α(t)*∆t
'''
def motion(x, u, dt):
    x[2] += u[1] * dt#theta
    x[0] += u[0] * math.cos(x[2]) * dt#x
    x[1] += u[0] * math.sin(x[2]) * dt#y
    x[3] = u[0]#v
    x[4] = u[1]#w
    return x

def predict_trajectory(x_init, v, w, config):
    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    #预测一系列的轨迹
    while time < config.predict_time:
        x = motion(x, [v, w], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory

#ob就是障碍物
#trajectory是几个时间片预测的绝对位置
def calc_obstacle_cost(trajectory, obstacle, config):
    ox = obstacle[:, 0]
    oy = obstacle[:, 1]

    #世界大小减去机器人的x和y
    m=min(config.board_length/2-abs(trajectory[-1,0]),config.board_width/2-abs(trajectory[-1,1]))
    if(m<config.robot_radius):
        return float("inf")
    
    #计算距离每个时间上与障碍的距离
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)#返回欧几里得距离

    #检测碰撞
    if config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius+config.obstacle_radius).any():
            return float("Inf")

    min_r = np.min(r)
    min_r=min(min_r,m)
    return 1000.0 / min_r  # OK


def calc_control_and_trajectory(x, dw, config, goal, obstacle):
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # best_speed=0
    # best_goal_cost=0
    # best_obstacle_cost=0
    # best_v=0
    # best_w=0
    bs=0;bg=0;bo=0;bw=0;bv=0

    #根据时间切片和两个速度的切片计算计算[v,w]的cost
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for w in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            trajectory = predict_trajectory(x_init, v, w, config)
            # cost的计算
            #G(v,ω)=σ(α*heading(v,ω)+β*dist(v,ω)+γ*vel(v,ω))
            #到目标距离越小越好,速度偏差越小越好,障碍物
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, obstacle, config)

            final_cost = to_goal_cost + speed_cost + ob_cost
            # 寻找最小的损失的路径
            if min_cost >= final_cost:
                bo=ob_cost;bg=to_goal_cost;bs=speed_cost;bv=v;bw=w
                min_cost = final_cost
                best_u = [v, w]
                best_trajectory = trajectory
                #速度太小了,应该要撞到障碍物了,这个时候就要偏移位置
                if abs(best_u[0]) < config.robot_stuck_flag_cons and abs(x[3]) < config.robot_stuck_flag_cons:
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u,best_trajectory

def move(x,u,ps):
    # print("initial ps",ps)
    d_theta=u[1]*config.predict_time
    v_left=(2*u[0]-config.dist_btw_wheels*u[1]/2)/2
    v_right=(2*u[0]+config.dist_btw_wheels*u[1]/2)/2

    start_time=robot.getTime()
    end_time=start_time

    right_motor.setVelocity(v_right)
    left_motor.setVelocity(v_left)
    while robot.step(timestep) != -1:
        right_motor.setVelocity(v_right)
        left_motor.setVelocity(v_left)
        end_time=robot.getTime()
        if(end_time-start_time>=config.predict_time-0.016):
            break

    _ps=[left_ps.getValue(),right_ps.getValue()]
    d_right=(_ps[1]-ps[1])*config.dist_per_radian
    d_left=(_ps[0]-ps[0])*config.dist_per_radian
    d_theta=(d_left-d_right)/config.dist_btw_wheels

    if(d_theta<0.0001):
        x[0]+=((d_right+d_left)/2)*math.cos(x[2])
        x[1]+=((d_right+d_left)/2)*math.sin(x[2])
    else:
        radius=(d_right/d_theta)+config.dist_btw_wheels/2
        x[0]+=radius*(math.sin(x[2]-d_theta)-math.sin(x[2]))
        x[1]+=radius*(math.cos(x[2]-d_theta)-math.cos(x[2]))
    x[2]-=d_theta
    x[3]=u[0]
    x[4]=u[1]
    return x,_ps

#将相对于机器人的坐标转为世界坐标
def transform(z,x,theta):
    #转化矩阵，参考教材
    m = [
    [np.cos(theta),-np.sin(theta),0],
    [np.sin(theta),np.cos(theta), 0],
    [0,            0,             1]
    ]
    val1= m[0][0]*z+m[0][1]*x
    val2 = m[1][0]*z+m[1][1]*x
    val3 = theta
    return [val1,val2]

#位置是100倍率
def main(ix=0,iy=0,theta=1.57,gx=2.2, gy=2.2, robot_type=RobotType.circle):

    max_speed=config.max_speed
    x = np.array([ix,iy,theta, 0.0, 0.0])

    ps=[0.0,0.0]
    goal = np.array([gx, gy])
    config.robot_type = robot_type
    trajectory = np.array(x)

    while robot.step(timestep) != -1:
        
        objects=camera.getRecognitionObjects()
        
        #z是第一个，x是第二个
        gps_val = gps.getValues()
        gps_val[2]*100#ix
        gps_val[0]*100#iy
        relativePos=[]
        
        #获取相对位置
        for obj in objects:
            #list_of_methods = inspect.getmembers(obj, predicate=inspect.ismethod)
            #print(list_of_methods)
            #print("pbj")
            p = obj.get_position()
            print(p)
            print(obj.get_id())
            print(-p[2],-p[0])
            relativePos.append([-p[2]*(100),-p[0]*(100)])
        
        #获取当前的偏航角
        theta = imu.getRollPitchYaw()[2]-0.4

        movingBarriers=[]
        for rpos in relativePos:
            movingBarriers.append(transform(rpos[0],rpos[1],theta=theta))
            
        for i in range(len(movingBarriers)):
            movingBarriers[i][0]+=gps_val[2]*100
            movingBarriers[i][1]+=gps_val[0]*100
        movingBarriers = np.array(movingBarriers)
        #print(movingBarriers)
        ob=config.ob
        if len(movingBarriers)!=0:
            ob = np.concatenate((ob,movingBarriers),axis=0)

        u, predicted_trajectory = dwa_control(x, config, goal, ob)

        x,ps= move(x,u,ps)

        g_val=gps.getValues()
        x[0]=g_val[2]*100
        x[1]=g_val[0]*100

        right_motor.setVelocity(0.0)
        left_motor.setVelocity(0.0)
        trajectory = np.vstack((trajectory, x))  
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        print(".\n")
        val = imu.getRollPitchYaw()
        print('imu val:',val)
        print()
        if dist_to_goal <= 2*config.robot_radius:
            print("Goal!!")
            break
    print("Done")

main(210,-200,1.57,100,220,RobotType.circle)



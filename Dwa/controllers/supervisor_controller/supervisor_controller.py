from controller import Supervisor

TIME_STEP = 32

robot = Supervisor()  # create Supervisor instance

# [CODE PLACEHOLDER 1]
node = robot.getFromDef('barrier')
node1 = robot.getFromDef('barrier1')

translation_field = node.getField('translation')
translation_field1 = node1.getField('translation')
i=0
left=0
right=1
new_value = [0, 0.1, -0.9]
dir=1

dir1=1
new_value1=[-1,0.1,-2.2]
max_z=-1.2
min_z = -2.2
while robot.step(TIME_STEP) != -1:
  # [CODE PLACEHOLDER 2]
    if new_value[0]>=right:
        dir=-1
    elif new_value[0]<=left:
        dir=1
    new_value[0]+=dir*0.0045
    
    if new_value1[2]>=max_z:
        dir1=-1
    elif new_value1[2]<=min_z:
        dir1=1
    new_value1[2]+=dir1*0.005
    
    translation_field.setSFVec3f(new_value)
    translation_field1.setSFVec3f(new_value1)



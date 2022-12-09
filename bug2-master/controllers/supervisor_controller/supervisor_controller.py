from controller import Supervisor

TIME_STEP = 32

robot = Supervisor()  # create Supervisor instance

# [CODE PLACEHOLDER 1]
node = robot.getFromDef('ball')
translation_field = node.getField('translation')

i=0
left=0
right=1
new_value = [-0.25, 0.0, 0.0]
dir=1
while robot.step(TIME_STEP) != -1:
  # [CODE PLACEHOLDER 2]
    if new_value[2]>=right:
        dir=-1
        
    elif new_value[2]<=left:
        dir=1
       
    new_value[2]+=dir*0.002
    
    translation_field.setSFVec3f(new_value)
    i += 1
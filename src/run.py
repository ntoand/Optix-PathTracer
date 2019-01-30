#/usr/bin/env python
import caveoptix

from omega import *
from cyclops import *
from omegaToolkit import *

co = caveoptix.initialize()
co.initOptix("data/diningroom.scene")

cam = getDefaultCamera()
cam.getController().setSpeed(1)
setNearFarZ(0.1, 1000)

mm = MenuManager.createAndInitialize()
menu = mm.getMainMenu()
mm.setMainMenu(menu)

#cmd = 'cam.setPosition(Vector3(' + str(campos[0]) + ',' + str(campos[1]) + ',' + str(campos[2]) + ')),' + \
#               'cam.setOrientation(Quaternion(' + str(camori[0]) + ',' + str(camori[1]) + ',' + str(camori[2]) + ',' + str(camori[3]) + '))'
menu.addButton("Go to camera 1", 'cam.setPosition(0, 0, 2.0)')

#cam.setPostion(7.0, 9.2, -6.0)

global needRedraw
needRedraw = False
def handleEvent():
    e = getEvent()
    global needRedraw
    if(e.isButtonDown(EventFlags.Left) or e.isButtonDown(EventFlags.Button5) or e.isButtonDown(EventFlags.Button7)): 
        needRedraw = True
    elif(e.isButtonUp(EventFlags.Left) or e.isButtonUp(EventFlags.Button5) or e.isButtonUp(EventFlags.Button7)):
        needRedraw = False
    
def onUpdate(frame, time, dt):
    global needRedraw
    if(needRedraw):
        co.resetDraw()

setEventFunction(handleEvent)
setUpdateFunction(onUpdate)
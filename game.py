from turtle import *
from freegames import square, vector
import random
import numpy as np
import redis
import json
import time

use_graphics = False

p1xy = vector(-100, 0)
p1aim = vector(4, 0)
p1body = set()

p2xy = vector(100, 0)
p2aim = vector(-4, 0)
p2body = set()

r = redis.Redis(host='localhost', port=6379, db=0)


def do_random_action(paim):
    if random.randint(0, 100) < 80:
        return paim
    else:
        rand_action = random.randint(0, 3)
        if rand_action == 0:
            return vector(0, 4)
        elif rand_action == 1:
            return vector(4, 0)
        elif rand_action == 2:
            return vector(0, -4)
        elif rand_action == 3:
            return vector(-4, 0)


p1MyBodies = []
p1OtherBodies = []
p1Actions = []
p1MyHeads = []

p2MyBodies = []
p2OtherBodies = []
p2Actions = []
p2MyHeads = []


def do_smart_action(paim, otherBodyVector, myBodyVectors, myHead, player):
    otherBody = np.zeros((200, 200))
    myBody = np.zeros((200, 200))
    for e in otherBodyVector:
        otherBody[abs(e.x)][abs(e.y)] = 1

    for e in myBodyVectors:
        myBody[abs(e.x)][abs(e.y)] = 1



    randint = str(random.randint(0, 100000000))
    r.set(randint, json.dumps({"myHead": [myHead.x/100, myHead.y/100],"otherBody": otherBody.tolist()}))
    r.rpush("jobs", randint)
    rand_action = None
    while rand_action is None:
        rand_action = r.get("completed:" + str(randint))
        if not rand_action:
            time.sleep(.01)
        else:
            r.delete("completed:" + randint)
            rand_action = json.loads(rand_action.decode('utf-8'))
            rand_action = np.argmax(rand_action)
    action = [0, 0, 0, 0, 0]
    action[rand_action] = 1



    if player == 'p1':
        p1MyBodies.append(myBody[::4,::4].tolist())
        p1OtherBodies.append(otherBody[::4,::4].tolist())
        p1Actions.append(action)
        p1MyHeads.append([myHead.x/100, myHead.y/100])
    elif player == 'p2':
        p2MyBodies.append(myBody[::4,::4].tolist())
        p2OtherBodies.append(otherBody[::4,::4].tolist())
        p2Actions.append(action)
        p2MyHeads.append([myHead.x/100, myHead.y/100])


    if rand_action == 0:
        return paim
    elif rand_action == 1:
        return vector(0, 4)
    elif rand_action == 2:
        return vector(4, 0)
    elif rand_action == 3:
        return vector(0, -4)
    elif rand_action == 4:
        return vector(-4, 0)


def inside(head):
    "Return True if head inside screen."
    return -200 < head.x < 200 and -200 < head.y < 200


def draw():
    global p1aim
    global p2aim
    "Advance players and draw game."
    p1xy.move(p1aim)
    p1head = p1xy.copy()

    p2xy.move(p2aim)
    p2head = p2xy.copy()

    if not inside(p1head) or p1head in p2body:
        print('Player blue wins! ' + str(len(p2body)))
        r.rpush("sample", json.dumps(
            {"myBody": p1MyBodies, "otherBody": p1OtherBodies, "action": p1Actions, "myHead": p1MyHeads,
             "winner": False}))
        r.rpush("sample", json.dumps(
            {"myBody": p2MyBodies, "otherBody": p2OtherBodies, "action": p2Actions, "myHead": p2MyHeads,
             "winner": True}))

        exit()
        return

    if not inside(p2head) or p2head in p1body:
        print('Player red wins! ' + str(len(p1body)))
        r.rpush("sample", json.dumps(
            {"myBody": p1MyBodies, "otherBody": p1OtherBodies, "action": p1Actions, "myHead": p1MyHeads,
             "winner": True}))
        r.rpush("sample", json.dumps(
            {"myBody": p2MyBodies, "otherBody": p2OtherBodies, "action": p2Actions, "myHead": p2MyHeads,
             "winner": False}))
        exit()
        return

    p1body.add(p1head)
    p2body.add(p2head)

    if use_graphics:
        square(p1xy.x, p1xy.y, 3, 'red')
        square(p2xy.x, p2xy.y, 3, 'blue')
        update()

        ontimer(draw, 50)

    p1aim = do_smart_action(p1aim, p1body, p2body, p1head, 'p1')
    p2aim = do_smart_action(p2aim, p2body, p1body, p2head, 'p2')

    if not use_graphics:
        draw()

    # p1aim = do_random_action(p1aim)
    # p2aim = do_random_action(p2aim)

if use_graphics:
    setup(420, 420, 370, 0)
    hideturtle()
    tracer(False)
    listen()

draw()
done()

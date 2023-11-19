
import socket
import time
import convert
import random

#観察空間の数：１(位置情報)
#行動空間の数：２(ダッシュ右移動、歩き右移動)
def state_num(): return 1
def action_num(): return 2

def sample(): return random.randint(0, action_num()-1)

def getIP(): return "192.168.XXX.XXX"


def reset(s):
    #操作、マリオリセット
    sendCommand(s, release_buttons())
    sendCommand(s, reset_x_pos()) 
    time.sleep(0.05)
    sendCommand(s, get_is_dead())
    time.sleep(0.05)
    return convert.reverse_hex(convert.recieve_data(s))

def sendCommand(s, content):
    content += '\r\n'
    s.sendall(content.encode())


def step(move, s):

    #コマンドに応じた操作
    if move == 0:
        print("move right")
        sendCommand(s, move_right())
    elif move == 1:
        print("jump")
        sendCommand(s, jump())
    done = False
    time.sleep(0.05)
    sendCommand(s, get_x_pos())
    x_pos = convert.reverse_hex(convert.recieve_data(s))
    time.sleep(0.02)
    sendCommand(s, get_is_dead())
    isDead = convert.reverse_hex(convert.recieve_data(s))
    print("マリオ死亡:", isDead)
    time.sleep(0.02)
    if isDead == 0:
        done = True
        return x_pos, done
    
    return x_pos, done
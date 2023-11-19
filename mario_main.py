import DQN as dqn
import socket
import time
import convert
import env

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((env.getIP(), 6000))


num_episode = 500  #学習エピソード数
memory_size = 50000  #replay bufferの大きさ
initial_memory_size = 500  #最初に貯めるランダムな遷移の数

episode_rewards = []
num_average_epidodes = 10
max_steps = 10000

prev_value = 0
stop_counter = 0

agent = dqn.DqnAgent(env.state_num(), env.action_num(), memory_size=memory_size)
state = env.reset(s)

#replay bufferにランダムな行動をしたときのデータを入れる
for step in range(initial_memory_size):
    
    action = env.sample() # ランダムに行動を選択
    next_state, done = env.step(action, s)
    reward = abs(next_state - state)
    print(next_state)

    #マリオが壁に当たった時など、停止したときはdoneをtrueにする
    if next_state != prev_value:
        stop_counter = 0
        prev_value = next_state
    else:
        stop_counter += 1
                
    if stop_counter >= 5:
        done = True           
        print("マリオ停止") 

    transition = {
        'state': state,
        'next_state': next_state,
        'reward': reward,
        'action': action,
        'done': int(done)
    }
    


    agent.replay_buffer.append(transition)
    state = env.reset(s) if done else next_state
    
#実際の訓練
for episode in range(num_episode):
    state = env.reset(s) 
    episode_reward = 0

    for t in range(max_steps):
        action = agent.get_action(state, episode)  # 行動を選択
        next_state, done = env.step(action, s)

        reward = abs(next_state - state)
        episode_reward += reward
        transition = {
            'state': state,
            'next_state': next_state,
            'reward': reward,
            'action': action,
            'done': int(done)
        }
        agent.replay_buffer.append(transition)
        agent.update_q()  # Q関数を更新
        state = next_state

        if done:
            break
    episode_rewards.append(episode_reward)
    if episode % 20 == 0:
        print("Episode %d finished | Episode reward %f" % (episode, episode_reward))
agent.save_qnet()
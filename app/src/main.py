import numpy as np
import matplotlib.pyplot as plt
from qlearning_agent import QLearningAgent
from grid_world import GridWorld

# 定数
#NB_EPISODE = 300   # エピソード数
EPSILON = .1        # 探索率
ALPHA = .1          # 学習率
GAMMA = .90         # 割引率
ACTIONS = np.arange(4)  # 行動の集合




""" Qテーブルの中身を表示する関数を作りたいnow """




if __name__ == '__main__':
    grid_env = GridWorld()  # 環境の初期化
    ini_state = grid_env.start_pos  # 初期状態
    
    # 既存のQテーブルがあれば再利用、なければ新規
    q_table_path = "saved_q_table.pkl"
    q_table_path = None
    
    
    
    agent = QLearningAgent(
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        actions=ACTIONS,
        observation=ini_state,
        q_table_path=q_table_path  # Qテーブルをロード
    )
    
    steps_to_goal = []  # ステップ数の記録
    is_end_episode = False

    # 実験開始
    NB_EPISODE = 0
    flg = 0
    
    
    while(True):
        step_count = 0
        while not is_end_episode:
            action = agent.act()  # 行動選択
            state, reward, is_end_episode = grid_env.step(action)  # 行動実行

            step_count += 1
            agent.observe(state, reward)  # 観測とQ値更新
        
        
        NB_EPISODE += 1
        steps_to_goal.append(step_count)
               

        # 環境のリセット
        state = grid_env.reset()
        agent.observe(state, step_penalty=0)
        is_end_episode = False
        
        
        
        
        if(step_count == 24): flg += 1
        else: flg = 0
        
        if(flg == 5): break
        
        
        
    q_table_path = "saved_q_table.pkl"

    # 学習結果のQテーブルを保存
    #agent.save_q_values(q_table_path)
    print(steps_to_goal)
    print(len(steps_to_goal))

    # 結果のプロット
    plt.plot(np.arange(NB_EPISODE), steps_to_goal)
    plt.xlabel("Episode")
    plt.ylabel("Steps to Goal")
    plt.title("Steps to reach the goal in each episode")
    plt.savefig("env0_copy.jpg")
    plt.show()

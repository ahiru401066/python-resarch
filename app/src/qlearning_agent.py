import copy
import numpy as np
import pickle


class QLearningAgent:
    """
        Q学習 エージェント
    """

    def __init__(
            self,
            alpha=.2,
            epsilon=.1,
            gamma=.99,
            actions=None,
            observation=None,
            q_table_path=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.state = str(observation)
        self.ini_state = str(observation)
        self.previous_state = None
        self.previous_action = None
        self.q_values = self._make_q_values(q_table_path)
        
        
        
    
    def save_q_values(self, file_path):
        """
        Qテーブルをファイルに保存
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.q_values, f)


    def load_q_values(self, file_path):
        """
        保存されたQテーブルをロード
        """
        with open(file_path, 'rb') as f:
            self.q_values = pickle.load(f)
        
        # 各キーと対応する配列を整形して表示   ロードは正常！！！
        """ for state, values in self.q_values.items():
            print(f"State {state}:")
            print(np.array2string(values, precision=2, separator=", "))
            print("-" * 40)  # 区切り線を表示
        """
    
    
    

    def _make_q_values(self, q_table_path):
        """ Qテーブルがある場合はロードする、ない場合は新規作成をする """
        
        self.q_table_path = q_table_path  # q_table_pathを属性として保存
    
        if self.q_table_path is not None:  # Qテーブルのパスが指定されている場合
            print("load!!!")
            self.load_q_values(q_table_path)  # Qテーブルをロード
        else: 
            print("not load!!!")
            # Q テーブルの初期化
            q_values = {}
            q_values[self.state] = np.repeat(0.0, len(self.actions))
            return q_values  # ここで返す
    
        return self.q_values  # Qテーブルをロードした場合、ロードした値を返す

    
    
    
    

    def init_state(self):
        """
            状態の初期化
        """
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state
    
    

    
    def act(self):
        
        # ε-greedy選択
        if np.random.uniform() < self.epsilon:  # random行動
            action = np.random.randint(0, len(self.q_values[self.state]))
        else:   # greedy 行動
            action = np.argmax(self.q_values[self.state])

        self.previous_action = action
        return action
    
    
        """
        行動選択メソッド。
        ε-greedy方式を使用して行動を選択
        一定の確率でランダムに行動を選択し、それ以外はソフトマックス法で選択
        """

    """
    #ε-greedy選択
        if np.random.uniform() < self.epsilon:  # ランダム行動
            action = np.random.randint(0, len(self.q_values[self.state]))
        else:  # ソフトマックスで行動選択
            # Q値からソフトマックスを計算して行動の確率を取得
            q_values = self.q_values[self.state]
            exp_q = np.exp(q_values - np.max(q_values))  # オーバーフロー対策
            probabilities = exp_q / np.sum(exp_q)  # 確率計算
    
            # 確率に基づいて行動を選択
            action = np.random.choice(len(q_values), p=probabilities)
    
        self.previous_action = action
        return action
    """

    

    

    def observe(self, next_state, step_penalty):
        """
            次の状態の観測とQ値の更新
        """
        next_state = str(next_state)
        if next_state not in self.q_values:  # 初めて訪れる状態であれば
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))

        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

        # ステップ数に応じたペナルティ（ステップ数を報酬の代わりに使用）
        self.learn(step_penalty)

    def learn(self, step_penalty):
        """
            Q値の更新
        """
        q = self.q_values[self.previous_state][self.previous_action]  # Q(s, a)
        max_q = max(self.q_values[self.state])  # max Q(s')
        # Q(s, a) = Q(s, a) + alpha*(ペナルティ + gamma*maxQ(s') - Q(s, a))
        self.q_values[self.previous_state][self.previous_action] = q + \
            (self.alpha * (step_penalty + (self.gamma * max_q) - q))

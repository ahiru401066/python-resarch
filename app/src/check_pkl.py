import pickle
import numpy as np

# .pklファイルを開いて読み込む
with open("saved_q_table__trash.pkl", "rb") as f:
    data = pickle.load(f)

# 各キーと対応する配列を整形して表示
for state, values in data.items():
    print(f"State {state}:")
    print(np.array2string(values, precision=2, separator=", "))
    print("-" * 40)  # 区切り線を表示

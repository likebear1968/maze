import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

class Board():
    DIRECTION = {
        0:-4, #上
        1:1,  #右
        2:4,  #下
        3:-1  #左
    }
    GOAL = 14

    def __init__(self):
        self.size = 4
        self.figure = plt.figure(figsize=(5, 5))
        self.line = self.create_maze()

    def create_maze(self):
        ax = plt.gca()
        # 壁
        plt.plot([1, 1], [1, 2], color='red', linewidth=2)
        plt.plot([1, 2], [3, 3], color='red', linewidth=2)
        plt.plot([2, 2], [3, 2], color='red', linewidth=2)
        plt.plot([2, 3], [2, 2], color='red', linewidth=2)
        plt.plot([1, 2], [1, 1], color='red', linewidth=2)
        plt.plot([2, 3], [1, 1], color='red', linewidth=2)
        plt.plot([3, 3], [3, 4], color='red', linewidth=2)
        # 状態
        plt.text(0.5, 3.5, 'S0', size=14, ha='center')
        plt.text(1.5, 3.5, 'S1', size=14, ha='center')
        plt.text(2.5, 3.5, 'S2', size=14, ha='center')
        plt.text(3.5, 3.5, 'S3', size=14, ha='center')
        plt.text(0.5, 2.5, 'S4', size=14, ha='center')
        plt.text(1.5, 2.5, 'S5', size=14, ha='center')
        plt.text(2.5, 2.5, 'S6', size=14, ha='center')
        plt.text(3.5, 2.5, 'S7', size=14, ha='center')
        plt.text(0.5, 1.5, 'S8', size=14, ha='center')
        plt.text(1.5, 1.5, 'S9', size=14, ha='center')
        plt.text(2.5, 1.5, 'S10', size=14, ha='center')
        plt.text(3.5, 1.5, 'S11', size=14, ha='center')
        plt.text(0.5, 0.5, 'S12', size=14, ha='center')
        plt.text(1.5, 0.5, 'S13', size=14, ha='center')
        plt.text(2.5, 0.5, 'S14', size=14, ha='center')
        plt.text(3.5, 0.5, 'S15', size=14, ha='center')
        plt.text(0.5, 3.3, 'START', ha='center')
        plt.text(2.5, 0.3, 'GOAL', ha='center')
        # 描画範囲の設定と目盛りを消す
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        # 開始位置に緑丸を描画
        line, = ax.plot([0.5], [3.5], marker="o", color='g', markersize=60)
        return line

    def play_movie(self, history):
        def init():
            self.line.set_data([], [])
            return (self.line,)
        def animate(i):
            state = history[i][0]
            x = (state % self.size) + 0.5
            y = 3.5 - int(state / self.size)
            self.line.set_data(x, y)
            return (self.line,)

        anime = animation.FuncAnimation(
            self.figure,
            animate,
            init_func=init,
            frames=len(history),
            interval=200,
            repeat=False
        )
        return HTML(anime.to_html5_video())

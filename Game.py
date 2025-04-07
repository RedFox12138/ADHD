import numpy as np
from psychopy import visual, core, event, sound
import random
import time

# 创建窗口（更高的刷新率设置）
win = visual.Window(color=(-0.8, -0.8, -0.8),
                    screen=0, waitBlanking=True)



# 准备期（不变）
def preparation_phase(duration=5):
    text = visual.TextStim(win, text="准备开始实验\n请保持放松", height=0.1)
    text.draw()
    win.flip()
    core.wait(duration)


# 静息阶段（不变）
def resting_phase(duration=60):
    images = ["lake.jpg", "clouds.jpg"]
    image_stims = [visual.ImageStim(win, image=img, size=3.0) for img in images]
    start_time = time.time()
    current_img = 0

    while time.time() - start_time < duration:
        image_stims[current_img].draw()
        win.flip()
        core.wait(10)
        current_img = 1 - current_img
        if event.getKeys(keyList=['escape']):
            win.close()
            core.quit()


# 休息阶段（不变）
def break_phase(duration=5):
    text = visual.TextStim(win, text="休息一下", height=0.1)
    text.draw()
    win.flip()
    core.wait(duration)


def attention_phase(duration=60):
    # 极限参数设置
    trial_duration = 0.5  # 500ms反应窗口
    target_prob = 0.3  # 目标概率
    blank_duration = 0.1  # 刺激间隔100ms

    # 精确计时器
    global_clock = core.Clock()
    rt_clock = core.Clock()

    # 刺激参数（精心设计的混淆刺激）
    colors = {
        'target': (0.9, -1, -1),  # 高饱和度红色
        'distractor1': (0.7, -0.8, -0.8),  # 低饱和度红
        'distractor2': (0.9, -0.5, -0.5),  # 粉红
        'distractor3': (0.9, -0.3, -1)  # 橙红
    }

    # 优化刺激呈现
    stim = visual.Circle(win, radius=0.15, edges=64)
    # fixation = visual.TextStim(win, text="+", color='white', height=0.1)
    mask = visual.Rect(win, width=0.4, height=0.4, fillColor=(-0.8, -0.8, -0.8))

    # 创建鼠标对象
    mouse = event.Mouse(win=win)

    # 性能计数器
    correct = 0
    false_alarms = 0
    misses = 0
    total_trials = 0

    # 主循环
    while global_clock.getTime() < duration:
        # 生成刺激
        is_target = random.random() < target_prob

        # 显示掩蔽（100ms）
        mask.draw()
        # fixation.draw()
        win.flip()
        core.wait(blank_duration)

        # 显示刺激
        if is_target:
            stim.fillColor = colors['target']
        else:
            stim.fillColor = random.choice(list(colors.values())[1:])

        stim.draw()
        # fixation.draw()
        win.flip()

        # 重置鼠标状态
        mouse.clickReset()

        # 反应检测（改为检测鼠标左键）
        response = False
        rt_clock.reset()
        while rt_clock.getTime() < trial_duration:
            if mouse.getPressed()[0]:  # 检测鼠标左键点击
                response = True
                rt = rt_clock.getTime() * 1000  # 毫秒
                break

        # 性能记录
        if is_target:
            if response:
                correct += 1
            else:
                misses += 1
        else:
            if response:
                false_alarms += 1

        total_trials += 1

        # 极短间隔（100ms）
        mask.draw()
        # fixation.draw()
        win.flip()
        core.wait(blank_duration)

    # 结果分析
    hit_rate = correct / (correct + misses) if (correct + misses) > 0 else 0
    false_alarm_rate = false_alarms / (total_trials - correct - misses) if (total_trials - correct - misses) > 0 else 0

    # 显示极简结果
    results = visual.TextStim(win,
                              text=f"命中率: {hit_rate * 100:.1f}%\n虚报率: {false_alarm_rate * 100:.1f}%",
                              color='white',
                              height=0.08
                              )
    results.draw()
    win.flip()
    core.wait(3)


# 主实验流程（不变）
def run_experiment():
    # print("准备阶段开始")
    # preparation_phase(5)
    #
    # print("静息阶段开始")
    # resting_phase(60)
    #
    # print("休息阶段开始")
    # break_phase(5)

    print("注意力阶段开始")
    attention_phase(10)

    text = visual.TextStim(win, text="实验结束，谢谢参与！", height=0.1)
    text.draw()
    win.flip()
    core.wait(3)
    win.close()


if __name__ == "__main__":
    run_experiment()
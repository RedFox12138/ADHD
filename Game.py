from psychopy import visual, core, event
import random
import time

# 创建全屏窗口
win = visual.Window(fullscr=True, monitor="testMonitor", units="norm")


# 准备期
def preparation_phase(duration=5):
    text = visual.TextStim(win, text="准备开始实验\n请保持放松", height=0.1)
    text.draw()
    win.flip()
    core.wait(duration)


# 静息阶段
def resting_phase(duration=60):
    # 准备两张图片交替显示
    images = ["lake.jpg", "clouds.jpg"]  # 需要准备这两张图片
    image_stims = [visual.ImageStim(win, image=img, size=3.0) for img in images]

    start_time = time.time()
    current_img = 0  # 当前显示的图片索引

    while time.time() - start_time < duration:
        # 显示当前图片
        image_stims[current_img].draw()
        win.flip()

        # 每3秒切换一次图片
        core.wait(3)
        current_img = 1 - current_img  # 在0和1之间切换

        # 检查是否要退出
        if event.getKeys(keyList=['escape']):
            win.close()
            core.quit()


# 休息阶段
def break_phase(duration=5):
    text = visual.TextStim(win, text="休息一下", height=0.1)
    text.draw()
    win.flip()
    core.wait(duration)


# 生成随机四位数加减法题目
def generate_math_problem():
    a = random.randint(1000, 9999)
    b = random.randint(1000, 9999)
    operation = random.choice(['+', '-'])

    if operation == '+':
        answer = a + b
    else:
        answer = a - b

    return f"{a} {operation} {b} = ?", answer


# 注意力集中阶段
def attention_phase(duration=60):
    start_time = time.time()
    problem_start_time = time.time()
    problem, answer = generate_math_problem()

    while time.time() - start_time < duration:
        # 每10秒更换题目
        if time.time() - problem_start_time >= 10:
            problem, answer = generate_math_problem()
            problem_start_time = time.time()

        # 显示题目
        text = visual.TextStim(win, text=problem, height=0.1)
        text.draw()
        win.flip()

        # 检查是否要退出
        if event.getKeys(keyList=['escape']):
            win.close()
            core.quit()

        core.wait(0.1)  # 短暂等待，减少CPU使用率


# 主实验流程
def run_experiment():
    # 准备期
    print("准备阶段开始")
    preparation_phase(5)

    # 静息阶段
    print("静息阶段开始")
    resting_phase(60)

    # 休息阶段
    print("休息阶段开始")
    break_phase(5)

    # 注意力集中阶段
    print("注意力阶段开始")
    attention_phase(60)

    # 实验结束
    text = visual.TextStim(win, text="实验结束，谢谢参与！", height=0.1)
    text.draw()
    win.flip()
    core.wait(3)
    win.close()


# 运行实验
if __name__ == "__main__":
    run_experiment()